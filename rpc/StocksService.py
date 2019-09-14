from rpc.protoc import stocks_pb2, stocks_pb2_grpc
import api


class StocksService(stocks_pb2_grpc.StocksServicer):
    def GetAllStocks(self, request, context):
        print("request: ", request)
        res = api.get_all_stocks()
        return stocks_pb2.GetAllStocksOutput(value=res)
