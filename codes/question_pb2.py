# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: question.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='question.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0equestion.proto\x1a\x0c\x63ommon.proto\"4\n\x0fQuestionRequest\x12\x0f\n\x07user_id\x18\x01 \x01(\x05\x12\x10\n\x08sequence\x18\x02 \x01(\x03\"\x94\x01\n\x10QuestionResponse\x12\x0f\n\x07user_id\x18\x01 \x01(\x05\x12\x10\n\x08sequence\x18\x02 \x01(\x03\x12\x19\n\x11has_next_question\x18\x03 \x01(\x08\x12\x0f\n\x07\x63\x61pital\x18\x04 \x01(\x01\x12\x1e\n\x08\x64\x61ilystk\x18\x05 \x03(\x0b\x32\x0c.DoubleArray\x12\x11\n\tpositions\x18\x06 \x03(\x01\x32?\n\x08Question\x12\x33\n\x0cget_question\x12\x10.QuestionRequest\x1a\x11.QuestionResponseb\x06proto3'
  ,
  dependencies=[common__pb2.DESCRIPTOR,])




_QUESTIONREQUEST = _descriptor.Descriptor(
  name='QuestionRequest',
  full_name='QuestionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='user_id', full_name='QuestionRequest.user_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sequence', full_name='QuestionRequest.sequence', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=84,
)


_QUESTIONRESPONSE = _descriptor.Descriptor(
  name='QuestionResponse',
  full_name='QuestionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='user_id', full_name='QuestionResponse.user_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sequence', full_name='QuestionResponse.sequence', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='has_next_question', full_name='QuestionResponse.has_next_question', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='capital', full_name='QuestionResponse.capital', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dailystk', full_name='QuestionResponse.dailystk', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='positions', full_name='QuestionResponse.positions', index=5,
      number=6, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=235,
)

_QUESTIONRESPONSE.fields_by_name['dailystk'].message_type = common__pb2._DOUBLEARRAY
DESCRIPTOR.message_types_by_name['QuestionRequest'] = _QUESTIONREQUEST
DESCRIPTOR.message_types_by_name['QuestionResponse'] = _QUESTIONRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

QuestionRequest = _reflection.GeneratedProtocolMessageType('QuestionRequest', (_message.Message,), {
  'DESCRIPTOR' : _QUESTIONREQUEST,
  '__module__' : 'question_pb2'
  # @@protoc_insertion_point(class_scope:QuestionRequest)
  })
_sym_db.RegisterMessage(QuestionRequest)

QuestionResponse = _reflection.GeneratedProtocolMessageType('QuestionResponse', (_message.Message,), {
  'DESCRIPTOR' : _QUESTIONRESPONSE,
  '__module__' : 'question_pb2'
  # @@protoc_insertion_point(class_scope:QuestionResponse)
  })
_sym_db.RegisterMessage(QuestionResponse)



_QUESTION = _descriptor.ServiceDescriptor(
  name='Question',
  full_name='Question',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=237,
  serialized_end=300,
  methods=[
  _descriptor.MethodDescriptor(
    name='get_question',
    full_name='Question.get_question',
    index=0,
    containing_service=None,
    input_type=_QUESTIONREQUEST,
    output_type=_QUESTIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_QUESTION)

DESCRIPTOR.services_by_name['Question'] = _QUESTION

# @@protoc_insertion_point(module_scope)
