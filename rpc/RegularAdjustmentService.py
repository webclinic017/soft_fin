from rpc.protoc import regular_adjustment_pb2_grpc
from rpc.protoc import regular_adjustment_pb2
import api


class RegularAdjustmentService(regular_adjustment_pb2_grpc.RegularAdjustmentServicer):

    def StockLeastPosition(self, request, context):
        res = api.stock_least_position(request.user_position, request.setting)
        return regular_adjustment_pb2.StockLeastPositionOutput(value=res)

    def StockVolatility(self, request, context):
        res = api.stock_volatility(request.stock_code, request.time, request.setting)
        return regular_adjustment_pb2.StockVolatilityOutput(value=res)

    def StockChange(self, request, context):
        res = api.stock_change(request.stock_code, request.time, request.top, request.bottom)
        return regular_adjustment_pb2.StockChangeOutput(value=res)

    def StockMeanReturn(self, request, context):
        res = api.stock_mean_return(request.stock_code, request.time, request.setting)
        return regular_adjustment_pb2.StockMeanReturnOutput(value=res)
