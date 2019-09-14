from rpc.protoc import option_futures_pb2, option_futures_pb2_grpc
import api


class OptionFuturesService(option_futures_pb2_grpc.OptionFuturesServicer):
    """
    =========================================================
    期权部分函数文档
    =========================================================
    """

    def PortfolioDelta(self, request, context):
        print("request: ", request)
        res = api.portfolio_delta(
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t)
        return option_futures_pb2.PortfolioDeltaOutput(value=res)

    def PortfolioGamma(self, request, context):
        print("request: ", request)
        res = api.portfolio_gamma(
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t)
        return option_futures_pb2.PortfolioGammaOutput(value=res)

    def PortfolioVega(self, request, context):
        print("request: ", request)
        res = api.portfolio_vega(
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t)
        return option_futures_pb2.PortfolioVegaOutput(value=res)

    def PortfolioTheta(self, request, context):
        print("request: ", request)
        res = api.portfolio_theta(
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t)
        return option_futures_pb2.PortfolioThetaOutput(value=res)

    def PortfolioRho(self, request, context):
        print("request: ", request)
        res = api.portfolio_rho(
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t)
        return option_futures_pb2.PortfolioRhoOutput(value=res)

    def PortfolioVolatility(self, request, context):
        print("request: ", request)
        res = api.portfolio_volatility(
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t)
        return option_futures_pb2.PortfolioVolatilityOutput(value=res)

    def PortfolioEarningRate(self, request, context):
        print("request: ", request)
        res = api.portfolio_earning_rate(
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t)
        return option_futures_pb2.PortfolioEarningRateOutput(value=res)

    def RetrainDeltaModel(self, request, context):
        print("request: ", request)

    def RetrainGammaModel(self, request, context):
        print("request: ", request)

    def FitDelta(self, request, context):
        print("request: ", request)

    def FitGamma(self, request, context):
        print("request: ", request)

    def CalOptionAmt(self, request, context):
        print("request: ", request)

    def GenerateRecommendOptionDelta(self, request, context):
        print("request: ", request)

    def GenerateRecommendOptionGamma(self, request, context):
        print("request: ", request)

    """
    =========================================================
    期货部分函数文档
    =========================================================
    """

    def PortfolioBeta(self, request, context):
        print("request: ", request)

    def RetrainBetaModel(self, request, context):
        print("request: ", request)

    def FitBeta(self, request, context):
        print("request: ", request)

    def CalFutureAmt(self, request, context):
        print("request: ", request)

    def GenerateRecommendFuture(self, request, context):
        print("request: ", request)
