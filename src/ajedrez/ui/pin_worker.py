"""Reuse the tested latest-position mailbox, cancellation and bounded cache."""
from .vulnerability_worker import VulnerabilityWorker
from ..pins import find_pins


class PinWorker(VulnerabilityWorker):
    def _calculate(self, board, known, cancelled):
        return find_pins(board, turn_known=known, cancelled=cancelled)
