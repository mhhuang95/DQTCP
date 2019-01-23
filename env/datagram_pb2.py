# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: datagram.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='datagram.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0e\x64\x61tagram.proto\"x\n\x04\x44\x61ta\x12\x0f\n\x07seq_num\x18\x01 \x01(\x07\x12\x0f\n\x07send_ts\x18\x02 \x01(\x07\x12\x12\n\nsent_bytes\x18\x03 \x01(\x06\x12\x16\n\x0e\x64\x65livered_time\x18\x04 \x01(\x07\x12\x11\n\tdelivered\x18\x05 \x01(\x06\x12\x0f\n\x07payload\x18\x06 \x01(\t\"y\n\x03\x41\x63k\x12\x0f\n\x07seq_num\x18\x01 \x01(\x07\x12\x0f\n\x07send_ts\x18\x02 \x01(\x07\x12\x12\n\nsent_bytes\x18\x03 \x01(\x06\x12\x16\n\x0e\x64\x65livered_time\x18\x04 \x01(\x07\x12\x11\n\tdelivered\x18\x05 \x01(\x06\x12\x11\n\tack_bytes\x18\x06 \x01(\x07\x62\x06proto3')
)




_DATA = _descriptor.Descriptor(
  name='Data',
  full_name='Data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='seq_num', full_name='Data.seq_num', index=0,
      number=1, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='send_ts', full_name='Data.send_ts', index=1,
      number=2, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sent_bytes', full_name='Data.sent_bytes', index=2,
      number=3, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='delivered_time', full_name='Data.delivered_time', index=3,
      number=4, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='delivered', full_name='Data.delivered', index=4,
      number=5, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='Data.payload', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=18,
  serialized_end=138,
)


_ACK = _descriptor.Descriptor(
  name='Ack',
  full_name='Ack',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='seq_num', full_name='Ack.seq_num', index=0,
      number=1, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='send_ts', full_name='Ack.send_ts', index=1,
      number=2, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sent_bytes', full_name='Ack.sent_bytes', index=2,
      number=3, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='delivered_time', full_name='Ack.delivered_time', index=3,
      number=4, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='delivered', full_name='Ack.delivered', index=4,
      number=5, type=6, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ack_bytes', full_name='Ack.ack_bytes', index=5,
      number=6, type=7, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=140,
  serialized_end=261,
)

DESCRIPTOR.message_types_by_name['Data'] = _DATA
DESCRIPTOR.message_types_by_name['Ack'] = _ACK
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Data = _reflection.GeneratedProtocolMessageType('Data', (_message.Message,), dict(
  DESCRIPTOR = _DATA,
  __module__ = 'datagram_pb2'
  # @@protoc_insertion_point(class_scope:Data)
  ))
_sym_db.RegisterMessage(Data)

Ack = _reflection.GeneratedProtocolMessageType('Ack', (_message.Message,), dict(
  DESCRIPTOR = _ACK,
  __module__ = 'datagram_pb2'
  # @@protoc_insertion_point(class_scope:Ack)
  ))
_sym_db.RegisterMessage(Ack)


# @@protoc_insertion_point(module_scope)
