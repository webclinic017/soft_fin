from rpc.protoc import conditional_trigger_pb2
from rpc.protoc import conditional_trigger_pb2_grpc
import api


class ConditionalTriggerService(conditional_trigger_pb2_grpc.ConditionalTriggerServicer):
    def StockMacd(self, request, context):
        print("request: ",request)
        res = api.stock_macd(request.stock_code, request.top, request.bottom)
        return conditional_trigger_pb2.StockMacdOutput(value=res)

    def StockRsi(self, request, context):
        print("request: ",request)
        res = api.stock_rsi(request.stock_code, request.rsi_time, request.top, request.bottom)
        return conditional_trigger_pb2.StockRsiOutput(value=res)

    def StockKdj(self, request, context):
        print("request: ",request)
        res = api.stock_kdj(request.stock_code, request.K, request.D, request.J)
        return conditional_trigger_pb2.StockKdjOutput(value=res)

    def StockRoc(self, request, context):
        print("request: ",request)
        res = api.stock_change(request.stock_code, request.time, request.top, request.bottom)
        return conditional_trigger_pb2.StockRocOutput(value=res)

    def StockSharpe(self, request, context):
        print("request: ",request)
        res = api.stock_sharpe(request.stock_code, request.setting)
        return conditional_trigger_pb2.StockSharpeOutput(value=res)
