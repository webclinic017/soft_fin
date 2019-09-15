from concurrent import futures
import time
import grpc
from rpc.protoc import regular_adjustment_pb2_grpc
from rpc.protoc import conditional_trigger_pb2_grpc
from rpc.protoc import adjustment_and_triggering_of_portfolio_pb2_grpc
from rpc.protoc import stocks_pb2_grpc
from rpc.protoc import option_futures_pb2_grpc
from rpc.protoc import citibank_api_pb2_grpc
from rpc import RegularAdjustmentService, ConditionalTriggerService, AdjustmentAndTriggeringOfPortfolioService, \
    StocksService, OptionFuturesService, CitibankApiService

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    regular_adjustment_pb2_grpc.add_RegularAdjustmentServicer_to_server(
        RegularAdjustmentService.RegularAdjustmentService(), server)
    conditional_trigger_pb2_grpc.add_ConditionalTriggerServicer_to_server(
        ConditionalTriggerService.ConditionalTriggerService(), server)
    adjustment_and_triggering_of_portfolio_pb2_grpc.add_AdjustmentAndTriggeringOfPortfolioServicer_to_server(
        AdjustmentAndTriggeringOfPortfolioService.AdjustmentAndTriggeringOfPortfolioService(), server)
    stocks_pb2_grpc.add_StocksServicer_to_server(
        StocksService.StocksService(), server)
    option_futures_pb2_grpc.add_OptionFuturesServicer_to_server(
        OptionFuturesService.OptionFuturesService(), server)
    citibank_api_pb2_grpc.add_CitibankApiServicer_to_server(
        CitibankApiService.CitibankApiService(), server
    )

    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
