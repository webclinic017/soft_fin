from rpc.protoc import stocks_pb2, stocks_pb2_grpc
import api


class StocksService(stocks_pb2_grpc.StocksServicer):
    def GetAllStocks(self, request, context):
        print("request: ", request)
        res = api.get_all_stocks()
        print("res: ", res)
        return stocks_pb2.GetAllStocksOutput(value=res)

    def GetStockHistory(self, request, context):
        print("request: ", request)
        output = api.get_stock_histroy(request.stock_code)
        res = []
        for row in output:
            r = stocks_pb2.GetStockHistoryRow(row=row)
            res.append(r)
        return stocks_pb2.GetStockHistoryOutput(value=res)
