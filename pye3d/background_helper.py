"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import multiprocessing as mp
import signal
from ctypes import c_bool

logger = logging.getLogger(__name__)


class EarlyCancellationError(Exception):
    pass


class Task_Proxy:
    """Future like object that runs a given generator in the background and returns is able to return the results incrementally"""

    def __init__(self, name, generator, args=(), kwargs={}, context=...):
        super().__init__()
        if context is ...:
            context = mp.get_context()

        self._should_terminate_flag = context.Value(c_bool, 0)
        self._completed = False
        self._canceled = False

        pipe_recv, pipe_send = context.Pipe(False)
        wrapper_args = self._prepare_wrapper_args(
            pipe_send, self._should_terminate_flag, generator
        )
        wrapper_args.extend(args)
        self.process = context.Process(
            target=self._wrapper, name=name, args=wrapper_args, kwargs=kwargs
        )
        self.process.daemon = True
        self.process.start()
        self.pipe = pipe_recv

    def _wrapper(self, pipe, _should_terminate_flag, generator, *args, **kwargs):
        """Executed in background, pipes generator results to foreground

        All exceptions are caught, forwarded to the foreground, and raised in
        `Task_Proxy.fetch()`. This allows users to handle failure gracefully
        as well as raising their own exceptions in the background task.
        """

        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught signal {sig} in:\n" + "".join(trace))
            # NOTE: Interrupt is handled in world/service/player which are responsible
            # for shutting down the background process properly

        signal.signal(signal.SIGINT, interrupt_handler)
        try:
            self._change_logging_behavior()
            logger.debug("Entering _wrapper")

            for datum in generator(*args, **kwargs):
                if _should_terminate_flag.value:
                    raise EarlyCancellationError("Task was cancelled")
                pipe.send(datum)
        except Exception as e:
            pipe.send(e)
            if not isinstance(e, EarlyCancellationError):
                import traceback

                logger.info(traceback.format_exc())
        else:
            pipe.send(StopIteration())
        finally:
            pipe.close()
            logger.debug("Exiting _wrapper")

    def _prepare_wrapper_args(self, *args):
        return list(args)

    def _change_logging_behavior(self):
        pass

    def fetch(self):
        """Fetches progress and available results from background"""
        if self.completed or self.canceled:
            return

        while self.pipe.poll(0):
            try:
                datum = self.pipe.recv()
            except EOFError:
                logger.debug("Process canceled be user.")
                self._canceled = True
                return
            else:
                if isinstance(datum, StopIteration):
                    self._completed = True
                    return
                elif isinstance(datum, EarlyCancellationError):
                    self._canceled = True
                    return
                elif isinstance(datum, Exception):
                    raise datum
                else:
                    yield datum

    def cancel(self, timeout=1):
        if not (self.completed or self.canceled):
            self._should_terminate_flag.value = True
            for x in self.fetch():
                # fetch to flush pipe to allow process to react to cancel comand.
                pass
        if self.process is not None:
            self.process.join(timeout)
            self.process = None

    @property
    def completed(self):
        return self._completed

    @property
    def canceled(self):
        return self._canceled
