# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import rpc.protoc.stocks_pb2 as stocks__pb2


class StocksStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.GetAllStocks = channel.unary_unary(
        '/Stocks/GetAllStocks',
        request_serializer=stocks__pb2.GetAllStocksInput.SerializeToString,
        response_deserializer=stocks__pb2.GetAllStocksOutput.FromString,
        )


class StocksServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def GetAllStocks(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_StocksServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'GetAllStocks': grpc.unary_unary_rpc_method_handler(
          servicer.GetAllStocks,
          request_deserializer=stocks__pb2.GetAllStocksInput.FromString,
          response_serializer=stocks__pb2.GetAllStocksOutput.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Stocks', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))