import unittest
import tinkoff_client


class TickBufferCapacityTests(unittest.TestCase):
    def test_buffer_retains_more_than_four_minutes_of_price_changes(self):
        quote = tinkoff_client._new_scalp_quote()
        self.assertGreaterEqual(quote["ticks"].maxlen, 2400)
