# coding: utf-8

import asyncio
import datetime as dt
import sys
import uuid
from contextlib import suppress
from typing import Callable, Dict, Optional


class Scheduler:
    def __init__(self):
        self.tasks: Dict[str, Periodic] = dict()

    async def add(
        self,
        func: Callable,
        name: Optional[str] = None,
        *,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        days: float = 0,
        weeks: float = 0,
        on_exception: Optional[Callable] = None,
        first_sleep: float = 1,
    ):
        name = name if name else str(uuid.uuid4())
        if name not in self.tasks:
            periodic_task = Periodic(
                func=func,
                name=name,
                seconds=seconds,
                minutes=minutes,
                hours=hours,
                days=days,
                weeks=weeks,
                on_exception=on_exception,
                first_sleep=first_sleep,
            )
            await periodic_task.start()
            self.tasks[name] = periodic_task

    async def stop(self):
        await asyncio.gather(*[task.stop() for task_name, task in self.tasks.items()])


class Periodic:
    """Edit by https://stackoverflow.com/a/37514633"""

    def __init__(
        self,
        func: Callable,
        name: str,
        *,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        days: float = 0,
        weeks: float = 0,
        on_exception: Optional[Callable] = None,
        first_sleep: float = 1,
    ):
        self.func = func
        self.name = name
        self.on_exception = on_exception
        self.timedelta: dt.timedelta = dt.timedelta(
            weeks=weeks,
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
        )
        self.first_sleep = first_sleep
        self.is_started = False
        self._task: asyncio.Task

    async def stop(self):
        if self.is_started:
            self.is_started = False
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                if isinstance(self._task, asyncio.Task):
                    await self._task

    async def start(self):
        if not self.is_started:
            self.is_started = True
            if sys.version_info[1] >= 11:
                # TaskGroup added in version 3.11.
                # async with asyncio.TaskGroup() as tg:  # type: ignore
                #     self._task = tg.create_task(self._run(), name=self.name)
                self._task = asyncio.create_task(self._run(), name=self.name)
            else:
                self._task = asyncio.create_task(self._run(), name=self.name)
        return self

    async def _run(self):
        if self.first_sleep and self.first_sleep > 0:
            await asyncio.sleep(self.first_sleep)
            self.first_sleep = 0

        try:
            await self._call_task()
            await asyncio.sleep(self.timedelta.total_seconds())
            self._task = asyncio.create_task(self._run(), name=self.name)
        except Exception as exc:
            if self.on_exception:
                if asyncio.iscoroutinefunction(self.on_exception):
                    await self.on_exception(exc)
                else:
                    self.on_exception(exc)

    async def _call_task(self):
        if asyncio.iscoroutinefunction(self.func):
            await self.func()
        else:
            await asyncio.to_thread(self.func)


def on_exception(exc: Exception):
    if sys.version_info[1] >= 11:
        raise exc
    else:
        asyncio.get_running_loop().stop()


async def __async_task():
    for i in range(3):
        print(f"test-{i+1}")
        await asyncio.sleep(1)
    # raise ValueError("Error")
