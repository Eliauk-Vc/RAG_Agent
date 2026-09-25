import asyncio
import unittest
from contextvars import ContextVar

from lightrag.utils import priority_limit_async_func_call


class QueueContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_queued_call_receives_its_own_request_context(self):
        request = ContextVar("test_request", default="none")

        @priority_limit_async_func_call(1)
        async def queued():
            await asyncio.sleep(0)
            return request.get()

        async def call(value):
            token = request.set(value)
            try:
                return await queued()
            finally:
                request.reset(token)

        try:
            self.assertEqual(await call("first"), "first")
            self.assertEqual(await call("second"), "second")
            self.assertEqual(await asyncio.gather(call("a"), call("b")), ["a", "b"])
            self.assertEqual(request.get(), "none")
        finally:
            await queued.shutdown()

    async def test_caller_timeout_cancels_provider_and_worker_remains_usable(self):
        started = asyncio.Event()
        stopped = asyncio.Event()

        @priority_limit_async_func_call(1)
        async def queued(slow):
            if slow:
                started.set()
                try:
                    await asyncio.sleep(20)
                finally:
                    stopped.set()
            return "ok"

        try:
            task = asyncio.create_task(queued(True))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(stopped.is_set())
            self.assertEqual(await asyncio.wait_for(queued(False), 1), "ok")
        finally:
            await queued.shutdown()
