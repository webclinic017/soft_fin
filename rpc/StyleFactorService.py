from rpc.protoc import style_factor_pb2_grpc, style_factor_pb2
import api


class StyleFactorService(style_factor_pb2_grpc.StyleFactorServicer):
    def GetSize(self, request, context):
        """语法：
        get_size(ts_code: str, fields: str)
        描述:
        获取指定股票的市值因子
        前置条件：
        ts_code为股票代码加交易所代号，如000001.SZ表示平安银行；
        fields限定为'circ_mv'和'total_mv',分别代表流通市值和总市值
        后置条件：
        返回对应股票的流通市值/总市值的自然对数
        """
        print("request: ", request)
        res = api.get_size(request.ts_code, request.fields)
        return style_factor_pb2.GetSizeOutput(value=res)

    def GetMomentum(self, request, context):
        """语法：
        get_momentum(ts_code: str)
        描述:
        获取指定股票的动量因子
        前置条件：
        ts_code为股票代码加交易所代号，如000001.SZ表示平安银行；
        后置条件：
        返回对应股票的动量因子
        """
        print("request: ", request)
        res = api.get_momentum(request.ts_code)
        return style_factor_pb2.GetMomentumOutput(value=res)

    def GetBookToMarket(self, request, context):
        """语法：
        get_book_to_market(stock_code: str)
        描述:
        获取指定股票的账面市值比因子
        前置条件：
        stock_code为股票代码，如300100表示双林股份；
        后置条件：
        返回对应股票的账面市值比因子
        """
        print("request: ", request)
        res = api.get_book_to_market(request.stock_code)
        return style_factor_pb2.GetBookToMarketOutput(value=res)

    def GetLiquidity(self, request, context):
        """语法：
        get_liquidity(stock_code: str)
        描述:
        获取指定股票的流动性因子
        前置条件：
        stock_code为股票代码加交易所代号，如000001.SZ表示平安银行；
        后置条件：
        返回对应股票的流动性因子
        """
        print("request: ", request)
        res = api.get_liquidity(request.stock_code)
        return style_factor_pb2.GetLiquidityOutput(value=res)
