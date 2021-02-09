# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: online_tracking_results.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='online_tracking_results.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1donline_tracking_results.proto\x1a\x0c\x63ommon.proto\"f\n\x06Tracks\x12\x16\n\x06tracks\x18\x01 \x03(\x0b\x32\x06.Track\x12&\n\x0e\x64\x65tection_type\x18\x02 \x01(\x0e\x32\x0e.DetectionType\x12\x1c\n\x14\x61ppearance_model_url\x18\x03 \x01(\t\"\x99\x01\n\x05Track\x12\x12\n\nconfidence\x18\x01 \x01(\x02\x12\x10\n\x08track_id\x18\x02 \x01(\t\x12%\n\x08\x66\x65\x61tures\x18\x03 \x01(\x0b\x32\x13.AppearanceFeatures\x12.\n\x12tracked_detections\x18\x04 \x03(\x0b\x32\x12.TrackingDetection\x12\x13\n\x0btracklet_id\x18\x05 \x01(\rb\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])




_TRACKS = _descriptor.Descriptor(
  name='Tracks',
  full_name='Tracks',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tracks', full_name='Tracks.tracks', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='detection_type', full_name='Tracks.detection_type', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='appearance_model_url', full_name='Tracks.appearance_model_url', index=2,
      number=3, type=9, cpp_type=9, label=1,
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
  serialized_start=47,
  serialized_end=149,
)


_TRACK = _descriptor.Descriptor(
  name='Track',
  full_name='Track',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='confidence', full_name='Track.confidence', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='track_id', full_name='Track.track_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='features', full_name='Track.features', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tracked_detections', full_name='Track.tracked_detections', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tracklet_id', full_name='Track.tracklet_id', index=4,
      number=5, type=13, cpp_type=3, label=1,
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
  serialized_start=152,
  serialized_end=305,
)

_TRACKS.fields_by_name['tracks'].message_type = _TRACK
_TRACKS.fields_by_name['detection_type'].enum_type = common__pb2._DETECTIONTYPE
_TRACK.fields_by_name['features'].message_type = common__pb2._APPEARANCEFEATURES
_TRACK.fields_by_name['tracked_detections'].message_type = common__pb2._TRACKINGDETECTION
DESCRIPTOR.message_types_by_name['Tracks'] = _TRACKS
DESCRIPTOR.message_types_by_name['Track'] = _TRACK
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Tracks = _reflection.GeneratedProtocolMessageType('Tracks', (_message.Message,), {
  'DESCRIPTOR' : _TRACKS,
  '__module__' : 'online_tracking_results_pb2'
  # @@protoc_insertion_point(class_scope:Tracks)
  })
_sym_db.RegisterMessage(Tracks)

Track = _reflection.GeneratedProtocolMessageType('Track', (_message.Message,), {
  'DESCRIPTOR' : _TRACK,
  '__module__' : 'online_tracking_results_pb2'
  # @@protoc_insertion_point(class_scope:Track)
  })
_sym_db.RegisterMessage(Track)


# @@protoc_insertion_point(module_scope)