from rpc.protoc import adjustment_and_triggering_of_portfolio_pb2
from rpc.protoc.adjustment_and_triggering_of_portfolio_pb2_grpc import AdjustmentAndTriggeringOfPortfolioServicer
import api


class AdjustmentAndTriggeringOfPortfolioService(AdjustmentAndTriggeringOfPortfolioServicer):
    def PortFolioVar(self, request, context):
        res = api.portfolio_var(request.portfolio, request.setting)
        return adjustment_and_triggering_of_portfolio_pb2.PortFolioOutput(value=res)

    def PortfolioVolatility(self, request, context):
        res = api.portfolio_volatility(request.portfolio, request.cash, request.setting)
        return adjustment_and_triggering_of_portfolio_pb2.PortFolioOutput(value=res)

    def PortfolioDiff(self, request, context):
        res = api.portfolio_diff(request.portfolio_id, request.portfolio, request.cash, request.alpha)
        return adjustment_and_triggering_of_portfolio_pb2.PortFolioOutput(value=res)
