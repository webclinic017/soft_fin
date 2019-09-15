from rpc.protoc import citibank_api_pb2_grpc
from rpc.protoc import citibank_api_pb2
import api


class CitibankApiService(citibank_api_pb2_grpc.CitibankApiServicer):
    def StockPoundage(self, request, context):
        print(request)
        res = api.stock_poundage(request.deal_amount)
        return citibank_api_pb2.StockPoundageOutput(value=res)

    def FuturePoundage(self, request, context):
        print(request)
        res = api.future_poundage(request.deal_amount)
        return citibank_api_pb2.FuturePoundageOutput(value=res)

    def OptionsPoundage(self, request, context):
        print(request)
        res = api.options_poundage(request.num_of_piece)
        return citibank_api_pb2.OptionsPoundageOutput(value=res)
