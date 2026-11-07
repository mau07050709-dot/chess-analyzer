"""Absolute pins and conservative, material-verified relative alignments."""
import unittest,json
from pathlib import Path
import chess
from ajedrez.pins import find_pins
from tests.test_vulnerability import position,kings


def absolute(symbol='N'):
    return position({'e1':'K','h8':'k','e2':symbol,'e8':'r'})


def relative(symbol='N'):
    # The rook is protected: the exposed queen cannot take it for free.
    return position({'h1':'K','d8':'k','e8':'r','e4':symbol,'e1':'Q'})


class PinTests(unittest.TestCase):
    def pin(self,board,square):
        return next((p for p in find_pins(board).pins if p.square==chess.parse_square(square)),None)

    def test_all_five_piece_types_can_be_absolutely_pinned_vertically(self):
        for symbol in 'PNBRQ':
            with self.subTest(symbol=symbol):
                self.assertEqual(self.pin(absolute(symbol),'e2').kind,'absolute')

    def test_horizontal_rook_pin(self):
        board=position({'a1':'K','h8':'k','c1':'N','h1':'r'})
        self.assertEqual(self.pin(board,'c1').kind,'absolute')

    def test_diagonal_bishop_pin(self):
        board=position({'a1':'K','h7':'k','c3':'N','h8':'b'})
        self.assertEqual(self.pin(board,'c3').kind,'absolute')

    def test_queen_can_pin_in_both_directions(self):
        for board,square in [(position({'a1':'K','h7':'k','c3':'N','h8':'q'}),'c3'),
                             (position({'e1':'K','h8':'k','e2':'N','e8':'q'}),'e2')]:
            self.assertEqual(self.pin(board,square).kind,'absolute')

    def test_relative_knight_bishop_rook_and_pawn_to_queen(self):
        for symbol in 'NBRP':
            with self.subTest(symbol=symbol):
                pin=self.pin(relative(symbol),'e4')
                self.assertIsNotNone(pin);self.assertEqual(pin.kind,'relative')
                self.assertEqual((pin.attacker,pin.target),(chess.E8,chess.E1))
                self.assertGreater(pin.gain_cp,0)

    def test_relative_bishop_to_rook_on_diagonal(self):
        board=position({'a1':'K','h8':'k','a8':'R','b7':'B','g2':'b'})
        pin=self.pin(board,'b7')
        self.assertEqual(pin.kind,'relative');self.assertEqual(pin.target,chess.A8)

    def test_relative_pawn_to_more_valuable_bishop(self):
        board=kings(a8='B',a7='P',a1='r',b1='n')
        # The exposed bishop cannot countercapture along the rook's file.
        pin=self.pin(board,'a7')
        self.assertIsNotNone(pin);self.assertEqual(pin.kind,'relative')

    def test_equal_or_less_valuable_target_is_not_a_relative_pin(self):
        for target in ('P','N','B'):
            board=relative('R');board.set_piece_at(chess.E1,chess.Piece.from_symbol(target))
            self.assertIsNone(self.pin(board,'e4'))

    def test_more_than_one_blocker_prevents_pin_to_king(self):
        board=absolute();board.set_piece_at(chess.E4,chess.Piece(chess.PAWN,True))
        self.assertIsNone(self.pin(board,'e2'))
        self.assertIsNone(self.pin(board,'e4'))

    def test_enemy_blocker_and_wrong_slider_direction_are_rejected(self):
        board=absolute();board.set_piece_at(chess.E4,chess.Piece(chess.PAWN,False))
        self.assertIsNone(self.pin(board,'e2'))
        for symbol in ('b','n','p','k'):
            board=absolute();board.set_piece_at(chess.E8,chess.Piece.from_symbol(symbol))
            self.assertIsNone(self.pin(board,'e2'))

    def test_direct_check_without_a_blocker_is_not_a_pin(self):
        board=absolute();board.remove_piece_at(chess.E2)
        self.assertEqual(find_pins(board).squares,frozenset())

    def test_piece_outside_the_attack_line_is_not_marked(self):
        board=absolute();board.remove_piece_at(chess.E2);board.set_piece_at(chess.D2,chess.Piece(chess.KNIGHT,True))
        self.assertIsNone(self.pin(board,'d2'))

    def test_absolute_pin_does_not_remove_legal_along_line_moves(self):
        board=absolute('R');fen=board.fen();legal=set(board.legal_moves)
        self.assertIn(chess.Move.from_uci('e2e3'),legal)
        self.assertEqual(self.pin(board,'e2').kind,'absolute')
        self.assertEqual(set(board.legal_moves),legal);self.assertEqual(board.fen(),fen)

    def test_target_can_capture_attacker_without_loss_so_relative_mark_is_omitted(self):
        board=relative();board.remove_piece_at(chess.D8);board.set_piece_at(chess.H8,chess.Piece(chess.KING,False))
        self.assertIsNone(self.pin(board,'e4'))

    def test_effective_defense_neutralizes_exposed_target(self):
        board=kings(d8='q',d5='P',d1='R',c1='R')
        # Queen would have to lose 9 for a rook; c1 recaptures on d1.
        self.assertIsNone(self.pin(board,'d5'))

    def test_pinned_attacker_cannot_create_a_relative_material_threat(self):
        board=position({'a1':'K','h8':'k','h7':'r','a7':'R','f7':'N','e7':'Q'})
        self.assertIsNone(self.pin(board,'f7'))

    def test_two_simultaneous_pins_both_colors_and_turn_independence(self):
        board=position({'e1':'K','e2':'N','e8':'r','h8':'k','h7':'n','h1':'R'})
        first=find_pins(board).squares
        self.assertEqual(first,frozenset([chess.E2,chess.H7]))
        board.turn=False
        self.assertEqual(find_pins(board).squares,first)
        self.assertEqual(find_pins(board,turn_known=False).squares,first)

    def test_mirrored_black_pin_has_the_same_meaning(self):
        for board,square in [(absolute(),chess.E2),(relative(),chess.E4)]:
            mirrored=board.mirror()
            original=next(p for p in find_pins(board).pins if p.square==square)
            actual=next(p for p in find_pins(mirrored).pins if p.square==square^56)
            self.assertEqual(actual.kind,original.kind)

    def test_absolute_classification_has_priority_over_a_relative_line(self):
        board=absolute();board.set_piece_at(chess.A2,chess.Piece(chess.ROOK,False))
        board.set_piece_at(chess.H2,chess.Piece(chess.QUEEN,True))
        pins=[p for p in find_pins(board).pins if p.square==chess.E2]
        self.assertEqual(len(pins),1);self.assertEqual(pins[0].kind,'absolute')

    def test_attacked_or_trapped_is_not_synonymous_with_pinned(self):
        for board in (kings(d4='N',d8='r'),kings(a8='N',a7='P',b7='P',b8='P')):
            self.assertEqual(find_pins(board).squares,frozenset())

    def test_remove_slider_removes_pin_immediately(self):
        board=absolute();self.assertIn(chess.E2,find_pins(board).squares)
        board.remove_piece_at(chess.E8);self.assertNotIn(chess.E2,find_pins(board).squares)

    def test_after_castling_and_historyless_resync_no_metadata_changes(self):
        board=chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 7 12');board.push_uci('e1g1')
        fen,stack=board.fen(en_passant='fen'),list(board.move_stack)
        find_pins(board);self.assertEqual(board.fen(en_passant='fen'),fen);self.assertEqual(board.move_stack,stack)
        visual=chess.Board(absolute().board_fen()+' w - - 0 1')
        self.assertIn(chess.E2,find_pins(visual,turn_known=False).squares)

    def test_budget_cancellation_and_invalid_positions_are_safe(self):
        self.assertEqual(find_pins(relative(),cancelled=lambda:True).squares,frozenset())
        self.assertFalse(find_pins(relative(),budget_ms=0).complete)
        invalid=absolute();invalid.remove_piece_at(chess.E1)
        self.assertEqual(find_pins(invalid).squares,frozenset())

    def test_user_reference_separates_absolute_pins_from_material_relative_criteria(self):
        from ajedrez.vulnerability import legal_exchange_gain
        ref=json.loads((Path(__file__).parent/'fixtures/pin_reference.json').read_text(encoding='utf-8'))
        for turn in ('w','b'):
            board=chess.Board(ref['placement']+f' {turn} - - 0 1')
            pins=find_pins(board,turn_known=False)
            self.assertEqual({chess.square_name(p.square):p.kind for p in pins.pins},
                             {'e6':'absolute','b7':'relative'})
        for blocker,capture,gain in [('c3','d2b4',300),('d4','d2d8',0)]:
            exposed=board.copy();exposed.remove_piece_at(chess.parse_square(blocker))
            exposed.turn=True
            self.assertEqual(legal_exchange_gain(exposed,chess.Move.from_uci(capture)),gain)


if __name__=='__main__':unittest.main()
