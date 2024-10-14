from asyncio import Queue
from enum import Enum
import sys, os
from typing import AsyncIterator, Dict, List, Optional, Tuple

import torch

from ktransformers.server.config.log import logger
from ktransformers.server.crud.assistants.assistants import AssistantDatabaseManager
from ktransformers.server.crud.assistants.messages import MessageDatabaseManager
from ktransformers.server.crud.assistants.runs import RunsDatabaseManager
from ktransformers.server.crud.assistants.threads import ThreadsDatabaseManager
from ktransformers.server.exceptions import request_error
from ktransformers.server.schemas.assistants.assistants import AssistantObject
from ktransformers.server.schemas.assistants.messages import MessageCreate, MessageObject, Role
from ktransformers.server.schemas.assistants.runs import RunObject
from ktransformers.server.schemas.assistants.threads import ThreadObject
from ktransformers.server.schemas.base import ObjectID, Order
from ktransformers.server.utils.multi_timer import Profiler


from .args import ConfigArgs, default_args


class BackendInterfaceBase:
    """
    Interface to inference frameworks. e.g. transformers, exllama.
    Implement __init__ and work
    """

    args: ConfigArgs
    profiler: Profiler = Profiler()

    def __init__(self, args: ConfigArgs = default_args):
        raise NotImplementedError

    async def inference(self, local_messages, request_unique_id: Optional[str]) -> AsyncIterator[str]:
        """
        work can be called directly, or by ThreadContext

        local_messages:
            When called by ThreadContext, local_messages are generated by ThreadContext.get_local_messages().
            Please deal with different local_messages
        request_unique_id:
            unique id of different requests, useful when using cache

        return:
            async str output for stream update

        """
        raise NotImplementedError

    def report_last_time_performance(self):
        try:
            tokenize_time = self.profiler.get_timer_sec("tokenize")
            prefill_time = self.profiler.get_timer_sec("prefill")
            decode_time = self.profiler.get_timer_sec("decode")
            prefill_count = self.profiler.get_counter("prefill")
            decode_count = self.profiler.get_counter("decode")

            logger.info(
                f"Performance(T/s): prefill {prefill_count/prefill_time}, decode {decode_count/decode_time}. Time(s):"
                f" tokenize {tokenize_time}, prefill {prefill_time}, decode {decode_time}"
            )
        except:
            logger.info(f"Performance statistics not recorded")


class ThreadContext:
    """
    A thread context holding assistant logics

    """

    args: ConfigArgs
    # Assistant Logic
    assistant: Optional[AssistantObject] = None
    related_threads: List[ThreadObject]
    thread: ThreadObject
    messages: List[MessageObject] = []
    run: RunObject

    interface: Optional[BackendInterfaceBase] = None

    queue: Optional[Queue] = None
    timer: Profiler = Profiler()

    def __init__(self, run: RunObject, interface: BackendInterfaceBase, args: ConfigArgs = default_args) -> None:
        self.args = args
        self.thread_manager = ThreadsDatabaseManager()
        self.message_manager = MessageDatabaseManager()
        self.runs_manager = RunsDatabaseManager()
        self.assistant_manager = AssistantDatabaseManager()
        self.thread = self.thread_manager.db_get_thread_by_id(run.thread_id)
        self.assistant = self.assistant_manager.db_get_assistant_by_id(run.assistant_id)
        self.messages = self.message_manager.db_list_messages_of_thread(run.thread_id, order=Order.ASC)
        logger.debug(f"{len(self.messages)} messages loaded from database")
        self.interface = interface
        self.update_by_run(run, args)

    def get_local_messages(self):
        """
        Get local messages, as the input to interface.work
        This function is intended to message preprocess e.g. apply chat template
        """
        raise NotImplementedError

    def update_by_run(self, run: RunObject, args: ConfigArgs = default_args):
        self.run = run
        self.args = args

    def put_user_message(self, message: MessageObject):
        assert (
            message.role.is_user()
            and message.thread_id == self.thread.id
            and message.status == MessageObject.Status.in_progress
        )
        self.messages.append(message)

    def delete_user_message(self, message_id: ObjectID):
        self.messages = [m for m in self.messages if m.id != message_id]

    async def work(self) -> AsyncIterator:
        logger.debug("start working")
        user_message = self.messages[-1]
        if not user_message.role.is_user():
            raise request_error("user must talk before LLM can talk")
        user_message.status = MessageObject.Status.completed
        user_message.sync_db()

        local_messages = self.get_local_messages()  # must get this before we interseted reply_message

        response_str_count = 0
        reply_message = self.message_manager.create_message_object(
            self.thread.id,
            self.run.id,
            MessageCreate(role=Role.assistant, content=""),
        )
        reply_message.assistant_id = self.assistant.id
        self.messages.append(reply_message)

        yield reply_message.stream_response_with_event(MessageObject.Status.created)
        yield reply_message.stream_response_with_event(MessageObject.Status.in_progress)
        yield self.run.stream_response_with_event(RunObject.Status.in_progress)

        async for token in self.interface.inference(local_messages, self.thread.id):
            if self.run.status == RunObject.Status.cancelling:
                logger.warn(f"Run {self.run.id} cancelling")
                break
            yield reply_message.append_message_delta(token)
            response_str_count += 1

        if self.run.status == RunObject.Status.cancelling:
            yield self.run.stream_response_with_event(RunObject.Status.cancelled)
            yield reply_message.stream_response_with_event(MessageObject.Status.incomplete)
        elif self.run.status == RunObject.Status.in_progress:
            yield self.run.stream_response_with_event(RunObject.Status.completed)
            yield reply_message.stream_response_with_event(MessageObject.Status.completed)
        else:
            raise NotImplementedError(f"{self.run.status} should not appear here")

        reply_message.sync_db()
        self.run.sync_db()
