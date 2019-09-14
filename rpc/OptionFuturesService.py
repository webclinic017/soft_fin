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

    def PortfolioGamma(self, request, context):
        print("request: ", request)

    def PortfolioVega(self, request, context):
        print("request: ", request)

    def PortfolioTheta(self, request, context):
        print("request: ", request)

    def PortfolioRho(self, request, context):
        print("request: ", request)

    def PortfolioVolatility(self, request, context):
        print("request: ", request)

    def PortfolioEarningRate(self, request, context):
        print("request: ", request)

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
