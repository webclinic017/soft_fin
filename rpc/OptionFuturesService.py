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
            request.asset_id, request.asset_amount, request.cash, request.begin_t, request.end_t, request.time)
        return option_futures_pb2.PortfolioEarningRateOutput(value=res)

    def RetrainDeltaModel(self, request, context):
        print("request: ", request)
        res = api.retrain_delta_model(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash, request.options
        )
        return option_futures_pb2.RetrainDeltaModelOutput(value=res)

    def RetrainGammaModel(self, request, context):
        print("request: ", request)
        res = api.retrain_gamma_model(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash, request.options1,
            request.options2
        )
        return option_futures_pb2.RetrainGammaModelOutput(value=res)

    def FitDelta(self, request, context):
        print("request: ", request)
        res = api.fit_delta(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash, request.options,
            request.begin_t, request.end_t
        )
        return option_futures_pb2.FitDeltaOutput(value=res)

    def FitGamma(self, request, context):
        print("request: ", request)
        res = api.fit_gamma(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash, request.options1,
            request.options2, request.begin_t, request.end_t
        )
        return option_futures_pb2.FitGammaOutput(value=res)

    def CalOptionAmt(self, request, context):
        print("request: ", request)
        res = api.cal_option_amt(
            request.total_value, request.option, request.portion, request.time
        )
        return option_futures_pb2.CalOptionAmtOutput(value=res)

    def GenerateRecommendOptionDelta(self, request, context):
        print("request: ", request)
        res = api.generate_recommend_option_delta(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash
        )
        return option_futures_pb2.GenerateRecommendOptionDeltaOutput(value=res)

    def GenerateRecommendOptionGamma(self, request, context):
        print("request: ", request)
        res = api.generate_recommend_option_gamma(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash
        )
        return option_futures_pb2.GenerateRecommendOptionGammaOutput(value=res)

    """
    =========================================================
    期货部分函数文档
    =========================================================
    """

    def PortfolioBeta(self, request, context):
        print("request: ", request)
        res = api.get_portfolio_beta(request.asset_id, request.weight)
        return option_futures_pb2.PortfolioBetaOutput(value=res)

    def RetrainBetaModel(self, request, context):
        print("request: ", request)
        res = api.retrain_beta_model(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash, request.futures
        )
        return option_futures_pb2.RetrainBetaModelOutput(value=res)

    def FitBeta(self, request, context):
        print("request: ", request)
        res = api.fit_beta(
            request.protfolio_id, request.asset_id, request.asset_amount, request.cash, request.futures
        )
        return option_futures_pb2.FitBetaOutput(value=res)

    def CalFutureAmt(self, request, context):
        print("request: ", request)
        res = api.cal_future_amt(
            request.total_value, request.futures, request.portion, request.begin_t
        )
        return option_futures_pb2.CalFutureAmtOutput(value=res)

    def GenerateRecommendFuture(self, request, context):
        print("request: ", request)
        res = api.generate_recommend_future(
            request.asset_id, request.asset_amount, request.cash, request.begin_t
        )
        return option_futures_pb2.GenerateRecommendFutureOutput(value=res)
