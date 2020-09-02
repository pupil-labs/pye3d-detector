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
import multiprocessing.connection
import signal
from ctypes import c_bool
from typing import Callable, Any, Dict, Tuple

logger = logging.getLogger(__name__)


class BackgroundProcess:
    class StoppedError(Exception):
        """Interaction with a BackgroundProcess that was stopped."""

    class NothingToReceiveError(Exception):
        """Trying to receive data from BackgroundProcess without sending input first."""

    class MultipleSendError(Exception):
        """Trying to send data without first receiving previous output."""

    def __init__(self, function: Callable):
        self._running = True
        self._busy = False

        self._pipe, remote_pipe = mp.Pipe(duplex=True)
        self._should_terminate_flag = mp.Value(c_bool, 0)

        self._process = mp.Process(
            name="Pye3D Background Process",
            daemon=True,
            target=BackgroundProcess._worker,
            kwargs=dict(
                function=function,
                pipe=remote_pipe,
                should_terminate_flag=self._should_terminate_flag,
            ),
        )
        self._process.start()

    @property
    def running(self) -> bool:
        """Whether background task is running (not necessarily doing work)."""
        return self._running

    @property
    def busy(self) -> bool:
        """Whether background task is doing work or ready to collect results."""
        return self._busy

    def send(self, *args: Tuple[Any], **kwargs: Dict[Any, Any]):
        """Send data to background process for processing.

        Raises MultipleSendError when called again without a call to recv() first.
        Raises StoppedError when called on a stopped process.
        """

        if not self.running:
            logger.error("Background process was closed previously!")
            raise BackgroundProcess.StoppedError()

        if self._busy:
            logger.error("Sending data without receiving previous output!")
            raise BackgroundProcess.MultipleSendError()

        self._pipe.send({"args": args, "kwargs": kwargs})
        self._busy = True

    def poll(self) -> bool:
        """Check if data is available for recv() from background task."""
        if not self.running:
            logger.error("Background process was closed previously!")
            raise BackgroundProcess.StoppedError()
        return self._pipe.poll()

    def recv(self) -> Any:
        """Returns results from background process.

        Blocks until results are available.

        Raises any Exception that occurred in backgrund process.
        Raises NothingToReceiveError when called without previous call to send().
        Raises StoppedError when called on a stopped process.
        """

        if not self.running:
            logger.error("Background process was closed previously!")
            raise BackgroundProcess.StoppedError()

        if not self._busy:
            logger.error("Querying background process without submitted data!")
            raise BackgroundProcess.NothingToReceiveError()

        try:
            results = self._pipe.recv()
        except EOFError:
            logger.error("Pipe was closed from background process!")
            raise BackgroundProcess.StoppedError()

        if isinstance(results, Exception):
            logger.error(f"Error in background process:\n{results}")
            raise results

        self._busy = False
        return results

    def cancel(self, timeout=-1):
        """Stop process as soon as current task is finished."""

        self._should_terminate_flag.value = 1
        if self.running:
            self._process.join(timeout)
        self._running = False

    @staticmethod
    def _install_sigint_interception():
        def interrupt_handler(sig, frame):
            import traceback

            trace = traceback.format_stack(f=frame)
            logger.debug(f"Caught (and dropping) signal {sig} in:\n" + "".join(trace))

        signal.signal(signal.SIGINT, interrupt_handler)

    @staticmethod
    def _worker(
        function: Callable,
        pipe: mp.connection.Connection,
        should_terminate_flag: mp.Value,
    ):
        # Intercept SIGINT (ctrl-c), do required cleanup in foreground process!
        BackgroundProcess._install_sigint_interception()

        while not should_terminate_flag.value:
            try:
                params = pipe.recv()
                args = params["args"]
                kwargs = params["kwargs"]
            except EOFError:
                logger.info("Pipe was closed from foreground process .")
                break

            try:
                results = function(*args, **kwargs)
            except Exception as e:
                pipe.send(e)
                logger.error(
                    f"Error executing background process with parameters {params}:\n{e}"
                )
                break

            pipe.send(results)
        else:
            logger.info("Background process received termination signal.")

        logger.info("Stopping background process.")
        pipe.close()
