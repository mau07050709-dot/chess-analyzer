"""Premove colors, projected layouts and legal execution/cancellation lifecycle."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import unittest
import chess
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage,QPainter,QColor,QPen
from tests.qt_test_support import test_application
from tests.test_last_move_highlight import cells_for,painted_highlight
from tests.image_helpers import reverse_cells
from tests.test_vulnerability import position
from ajedrez.premove import (detect_premove_cells,interpret_premove,PremoveMonitor,
                             plausible_intent,preview_placements)
from ajedrez.recognition import RecognitionResult,SquarePrediction,recognize_position
from ajedrez.live_board import LiveBoardTracker
from ajedrez.last_move_highlight import detect_highlights

SIMPLE='7k/p7/8/8/8/8/7P/K7 b - - 0 1'
RED=(180,96,66)


def observation(board,orientation='white'):
    return RecognitionResult(board.board_fen(),orientation,())


def ghost(board,move,*,castle=True,ep=False):
    output=board.copy();piece=output.remove_piece_at(move.from_square)
    output.set_piece_at(move.to_square,chess.Piece(move.promotion or piece.piece_type,piece.color))
    if castle and piece.piece_type==chess.KING and abs(move.to_square-move.from_square)==2:
        rank=chess.square_rank(move.from_square);short=move.to_square>move.from_square
        output.remove_piece_at(chess.square(7 if short else 0,rank))
        output.set_piece_at(chess.square(5 if short else 3,rank),chess.Piece(chess.ROOK,piece.color))
    if ep:output.remove_piece_at(move.to_square+(-8 if piece.color else 8))
    return output


class Flow:
    def __init__(self,board,orientation='white'):
        self.tracker=LiveBoardTracker(board);self.monitor=PremoveMonitor();self.orientation=orientation
    def feed(self,visual,red=None,stable=True):
        result=observation(visual,self.orientation)
        evidence=interpret_premove(result,red,self.tracker.board,self.monitor.pending,
                  turn_known=self.tracker.turn_known,orientation=self.orientation,epoch=self.monitor.revision,scope=(7,0))
        assert self.monitor.consume(evidence,self.tracker.board,visual_stable=stable)
        if evidence.hold:return evidence,None
        update=self.tracker.observe(evidence.interpreted,visual_stable=stable)
        self.monitor.after_tracking(self.tracker.board,visual_stable=stable)
        return evidence,update
    def twice(self,visual,red=None):
        self.feed(visual,red);return self.feed(visual,red)
    def pending(self,move):
        view=ghost(self.tracker.board,move)
        self.twice(view,cells_for(move,self.orientation));return view


class PremoveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=test_application()

    def test_real_image_red_cells_and_piece_projection_are_identified(self):
        image=QImage(str(Path(__file__).parent/'fixtures/premove_real_h2_g3.png'))
        red=detect_premove_cells(image)
        self.assertEqual(set(red.squares('white')),{chess.H2,chess.G3})
        recognized=recognize_position(image)
        self.assertEqual(recognized.uncertain_squares,())
        self.assertEqual(chess.Board(recognized.piece_placement+' w - - 0 1').piece_at(chess.G3),chess.Piece(chess.PAWN,True))
        yellow=detect_highlights(image)
        if yellow: self.assertTrue(set(red.squares('white')).isdisjoint(yellow.squares('white')))

    def test_real_red_detection_in_both_views_and_resolutions(self):
        image=QImage(str(Path(__file__).parent/'fixtures/premove_real_h2_g3.png'))
        for view,pixels in [('white',image),('black',reverse_cells(image))]:
            for side in (256,480,800):
                h=detect_premove_cells(pixels.scaled(side,side))
                with self.subTest(view=view,side=side):
                    self.assertIsNotNone(h);self.assertEqual(set(h.squares(view)),{chess.H2,chess.G3})

    def test_white_black_premoves_tolerant_colors_and_covered_centers(self):
        for board,uci in [(chess.Board(SIMPLE),'h2h3'),(chess.Board(SIMPLE).mirror(),'h7h6')]:
            move=chess.Move.from_uci(uci);visual=ghost(board,move)
            for view in ('white','black'):
                for color in [(174,90,57),(194,105,75),(150,66,51)]:
                    image=painted_highlight(visual,(move.from_square,move.to_square),view,color)
                    found=detect_premove_cells(image)
                    self.assertIsNotNone(found);self.assertEqual(set(found.squares(view)),{move.from_square,move.to_square})

    def test_yellow_one_red_three_red_and_red_board_theme_are_not_premoves(self):
        board=chess.Board()
        self.assertIsNone(detect_premove_cells(painted_highlight(board,(chess.E2,chess.E4))))
        for squares in [(chess.E2,),(chess.E2,chess.E4,chess.D4),tuple(chess.SQUARES)]:
            self.assertIsNone(detect_premove_cells(painted_highlight(board,squares,color=RED)))

    def test_red_arrow_or_corner_cross_does_not_become_two_full_square_marks(self):
        image=painted_highlight(chess.Board(),(),color=RED)
        painter=QPainter(image);painter.setPen(QPen(QColor(*RED),12))
        painter.drawLine(350,650,350,450);painter.drawLine(315,615,335,635)
        painter.drawLine(335,615,315,635);painter.end()
        self.assertIsNone(detect_premove_cells(image))

    def test_pending_does_not_change_turn_history_fen_or_rights(self):
        board=chess.Board('r3k2r/8/8/8/8/8/7P/R3K2R b KQkq - 17 12')
        flow=Flow(board);flow.pending(chess.Move.from_uci('h2h3'))
        self.assertEqual(flow.tracker.board.fen(en_passant='fen'),board.fen(en_passant='fen'))
        self.assertEqual(flow.tracker.board.move_stack,board.move_stack)
        self.assertTrue(flow.tracker.turn_known);self.assertEqual(flow.monitor.state,'PENDING')

    def test_executed_premove_recovers_two_plies_without_extra_move(self):
        board=chess.Board(SIMPLE);flow=Flow(board);move=chess.Move.from_uci('h2h3');flow.pending(move)
        final=board.copy();final.push_uci('a7a6');final.push(move)
        flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen())
        self.assertEqual(flow.monitor.last_outcome,'COMPLETED');self.assertIsNone(flow.monitor.pending)

    def test_opponents_move_is_confirmed_even_while_red_and_ghost_remain(self):
        board=chess.Board(SIMPLE);flow=Flow(board);move=chess.Move.from_uci('h2h3');flow.pending(move)
        opponent=board.copy();opponent.push_uci('a7a6')
        flow.twice(ghost(opponent,move),cells_for(move))
        self.assertEqual(flow.tracker.board.fen(),opponent.fen())
        self.assertIsNotNone(flow.monitor.pending)
        final=opponent.copy();final.push(move);flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(flow.monitor.last_outcome,'COMPLETED')

    def test_canceled_with_opponents_move_keeps_only_that_real_move(self):
        board=chess.Board(SIMPLE);flow=Flow(board);flow.pending(chess.Move.from_uci('h2h3'))
        final=board.copy();final.push_uci('a7a6');flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen())
        self.assertEqual(flow.monitor.last_outcome,'CANCELED');self.assertEqual(len(flow.tracker.board.move_stack),1)

    def test_manual_cancellation_is_not_a_move_and_does_not_change_turn(self):
        board=chess.Board(SIMPLE);flow=Flow(board);flow.pending(chess.Move.from_uci('h2h3'))
        flow.twice(board)
        self.assertEqual(flow.monitor.last_outcome,'CANCELED');self.assertEqual(flow.monitor.squares,())
        self.assertEqual(flow.tracker.board.fen(),board.fen());self.assertEqual(flow.tracker.board.move_stack,[])

    def test_replacement_keeps_only_latest_intent_and_stale_evidence_is_rejected(self):
        board=position({'a1':'K','h8':'k','g1':'N','a7':'p'},turn=False);flow=Flow(board)
        first=chess.Move.from_uci('g1f3');second=chess.Move.from_uci('g1h3')
        flow.pending(first)
        stale=interpret_premove(observation(ghost(board,first)),cells_for(first),board,flow.monitor.pending,
                                 epoch=flow.monitor.revision)
        flow.pending(second)
        self.assertEqual(flow.monitor.pending.intent.move,second)
        self.assertFalse(flow.monitor.consume(stale,board,visual_stable=True))
        self.assertEqual(flow.monitor.pending.intent.move,second)
        self.assertEqual(flow.tracker.board.fen(),board.fen())

    def test_disappearing_red_alone_cannot_confirm_the_unchanged_ghost(self):
        board=chess.Board(SIMPLE);flow=Flow(board);move=chess.Move.from_uci('h2h3');visual=flow.pending(move)
        for _ in range(4):
            evidence,_=flow.feed(visual)
            self.assertTrue(evidence.hold)
        self.assertEqual(flow.tracker.board.fen(),board.fen());self.assertTrue(flow.tracker.turn_known)

    def test_red_without_relocated_piece_is_still_an_intent_not_a_move(self):
        board=chess.Board(SIMPLE);flow=Flow(board);move=chess.Move.from_uci('h2h3')
        flow.twice(board,cells_for(move))
        self.assertEqual(flow.monitor.pending.intent.move,move)
        self.assertEqual(flow.tracker.board.fen(),board.fen())

    def test_black_premove_and_both_orientations_keep_turn_separate(self):
        for view in ('white','black'):
            board=chess.Board(SIMPLE).mirror();flow=Flow(board,view);move=chess.Move.from_uci('h7h6')
            flow.pending(move);self.assertEqual(flow.tracker.board.turn,chess.WHITE)
            final=board.copy();final.push_uci('a2a3');final.push(move);flow.twice(final)
            self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(flow.tracker.board.turn,chess.WHITE)

    def test_pending_capture_does_not_delete_its_victim_from_logical_board(self):
        board=position({'a1':'K','h8':'k','e4':'P','d5':'p','a7':'p'},turn=False)
        flow=Flow(board);move=chess.Move.from_uci('e4d5');flow.pending(move)
        self.assertEqual(flow.tracker.board.piece_at(chess.D5),chess.Piece(chess.PAWN,False))
        final=board.copy();final.push_uci('a7a6');final.push(move);flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(flow.monitor.last_outcome,'COMPLETED')

    def test_source_captured_cancels_premove_but_preserves_the_capture(self):
        board=position({'a1':'K','h8':'k','c3':'N','b4':'b','a7':'p'},turn=False)
        flow=Flow(board);move=chess.Move.from_uci('c3d5');flow.pending(move)
        final=board.copy();final.push_uci('b4c3')
        flow.twice(final,cells_for(move))
        self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(flow.monitor.last_outcome,'CANCELED')

    def test_changed_destination_occupancy_is_reconstructed_from_a_legal_opponent_move(self):
        board=position({'a1':'K','h8':'k','c3':'N','d8':'r'},turn=False)
        flow=Flow(board);move=chess.Move.from_uci('c3d5');flow.pending(move)
        opponent=board.copy();opponent.push_uci('d8d5')
        flow.twice(ghost(opponent,move),cells_for(move))
        self.assertEqual(flow.tracker.board.fen(),opponent.fen())
        self.assertEqual(flow.tracker.board.piece_at(chess.D5),chess.Piece(chess.ROOK,False))
        final=opponent.copy();final.push(move);flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen())

    def test_promotion_is_not_recorded_until_real_execution(self):
        board=chess.Board('1r5k/P7/8/8/8/8/8/7K b - - 0 1')
        flow=Flow(board);move=chess.Move.from_uci('a7b8q');flow.pending(move)
        self.assertEqual(flow.tracker.board.piece_at(chess.A7),chess.Piece(chess.PAWN,True))
        final=board.copy();final.push_uci('h8h7');final.push(move);flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(flow.monitor.last_outcome,'COMPLETED')

    def test_castle_premoves_both_sides_and_colors_preserve_rights_until_executed(self):
        for color in (True,False):
            for wing in ('c','g'):
                rank='1' if color else '8';turn='b' if color else 'w'
                board=chess.Board(f'r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1')
                flow=Flow(board);move=chess.Move.from_uci(f'e{rank}{wing}{rank}')
                flow.pending(move);self.assertEqual(flow.tracker.board.castling_xfen(),'KQkq')
                final=board.copy();final.push_uci('a8a7' if color else 'a1a2');final.push(move);flow.twice(final)
                self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(flow.monitor.last_outcome,'COMPLETED')

    def test_king_only_castle_preview_is_not_a_real_half_castle(self):
        board=chess.Board('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1');flow=Flow(board)
        move=chess.Move.from_uci('e1g1');flow.twice(ghost(board,move,castle=False),cells_for(move))
        self.assertEqual(flow.tracker.board.fen(),board.fen())

    def test_anticipatory_en_passant_uses_only_the_real_new_right(self):
        board=position({'h1':'K','h8':'k','e5':'P','d7':'p'},turn=False)
        flow=Flow(board);move=chess.Move.from_uci('e5d6');flow.pending(move)
        self.assertIsNone(flow.tracker.board.ep_square)
        opponent=board.copy();opponent.push_uci('d7d5')
        # An unobscured observation proves d7d5 before the EP ghost hides d5.
        flow.twice(opponent,cells_for(move))
        self.assertEqual(flow.tracker.board.ep_square,chess.D6)
        flow.twice(ghost(opponent,move,ep=True),cells_for(move))
        final=opponent.copy();final.push(move);flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(flow.monitor.last_outcome,'COMPLETED')

    def test_missing_en_passant_right_is_never_fabricated(self):
        board=position({'h1':'K','h8':'k','e5':'P','d5':'p'},turn=False);flow=Flow(board)
        move=chess.Move.from_uci('e5d6');flow.pending(move)
        for _ in range(2):flow.feed(ghost(board,move,ep=True),cells_for(move))
        self.assertIsNone(flow.tracker.board.ep_square);self.assertEqual(flow.tracker.board.fen(),board.fen())

    def test_hidden_pawn_cannot_prove_whether_opponent_played_d6_or_d5(self):
        board=position({'h1':'K','h8':'k','e5':'P','d7':'p'},turn=False)
        flow=Flow(board);move=chess.Move.from_uci('e5d6');flow.pending(move)
        opponent=board.copy();opponent.push_uci('d7d5')
        evidence,_=flow.twice(ghost(opponent,move,ep=True),cells_for(move))
        self.assertTrue(evidence.hold)
        self.assertEqual(flow.tracker.board.fen(),board.fen())
        # Once the pieces are visible, ordinary legal tracking proves both plies.
        final=opponent.copy();final.push(move);flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen())

    def test_check_that_invalidates_premove_keeps_only_real_checking_move(self):
        from tests.test_premove_regression import BASE
        board=chess.Board(BASE);flow=Flow(board);flow.pending(chess.Move.from_uci('h2h3'))
        final=board.copy();final.push_uci('b4d2');self.assertTrue(final.is_check())
        flow.twice(final);self.assertEqual(flow.tracker.board.fen(),final.fen())
        self.assertEqual(flow.monitor.last_outcome,'CANCELED')

    def test_three_real_plies_after_execution_use_existing_bounded_recovery(self):
        board=position({'a1':'K','h8':'k','a7':'n','h2':'P'},turn=False)
        flow=Flow(board);move=chess.Move.from_uci('h2h3');flow.pending(move)
        final=board.copy()
        for uci in ('a7b5','h2h3','b5c7'):final.push_uci(uci)
        flow.twice(final)
        self.assertEqual(flow.tracker.board.fen(),final.fen());self.assertEqual(len(flow.tracker.board.move_stack),3)

    def test_low_confidence_cannot_promote_masked_pixels_to_confirmed_truth(self):
        board=chess.Board(SIMPLE);move=chess.Move.from_uci('h2h3');visual=ghost(board,move)
        raw=RecognitionResult(visual.board_fen(),'white',(SquarePrediction(chess.H3,'P','P',.2),))
        evidence=interpret_premove(raw,cells_for(move),board)
        self.assertTrue(evidence.hold);self.assertIsNone(evidence.interpreted)

    def test_real_turn_red_move_is_not_vetoed_merely_for_its_color(self):
        board=chess.Board();move=chess.Move.from_uci('e2e4');final=board.copy();final.push(move)
        evidence=interpret_premove(observation(final),cells_for(move),board)
        self.assertFalse(evidence.hold);self.assertIsNone(evidence.intent)
        self.assertEqual(evidence.interpreted.piece_placement,final.board_fen())


if __name__=='__main__':unittest.main()
