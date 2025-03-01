from typing import Any, Optional
import numpy as np
import pandas as pd
import akshare as ak
import calendar
import datetime
from datetime import date, timedelta

from vnpy_ctastrategy.template import CtaTemplate
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy.trader.constant import Interval, Offset
from vnpy_ctastrategy.base import StopOrder

class Cpv4Cta(CtaTemplate):
    """
    这个示例策略演示了对日内持仓量 (OI) 进行拆分，以区分 T+0 和 T+1 交易者的持仓变化。
    然后基于调整后的持仓量，计算一个“价量相关系数”（PV），再结合 DOV 指标判断当日交易者
    类型，从而判断 PV 信号在次日的有效性或是否要反转。
    """

    # 策略参数
    dov_mean_lookback: int = 30        # DOV均值统计的回看天数
    dov_std_lookback: int = 50         # DOV标准差统计的回看天数
    dov_mean_percentile: int = 85      # DOV均值触发反转信号的百分位
    dov_std_percentile: int = 80       # DOV标准差触发反转信号的百分位
    mean_standard: int = 5            # 用于计算最近几天均值的窗口
    stop_loss_pct: float = 0.07       # 日内止损阈值
    take_profit_pct: float = 0.1      # 日内止盈阈值

    parameters = [
        "dov_mean_lookback",
        "dov_std_lookback",
        "dov_mean_percentile",
        "dov_std_percentile",
        "mean_standard",
        "stop_loss_pct",
        "take_profit_pct"
    ]

    # 策略运行时的变量
    fixed_size: int = 0
    variables = [
        'fixed_size'
    ]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """
        构造函数：在此进行一些初始化工作，比如定义用来存储 OI、成交量、收盘价等的列表。
        """
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 用于生成不同周期的 Bar（如果需要的话）
        self.bg = BarGenerator(self.on_bar)

        # 用于存储 K 线数据，如收盘价数组等
        self.am = ArrayManager()

        # 买卖/开平相关价格
        self.buy_price: float = 0.0
        self.sell_price: float = 0.0
        self.short_price: float = 0.0
        self.cover_price: float = 0.0

        # 仓位控制
        self.fixed_size = 1
        self.pos = 0

        # 存储上一根 Bar，方便计算差值
        self.last_bar: Optional[BarData] = None

        # 日内跟踪用的变量
        self.daily_volume: float = 0.0
        self.oi_minu_change: list = []
        self.oi_daily_change: float = 0.0
        self.volume_list: list = []
        self.vol_minu_change: list = []
        self.close_list: list = []
        self.w_list: list = []
        self.oi_change_t1: list = []
        self.oi_change_t0: list = []
        self.oi_adj_value: list = []
        self.pv_dic: dict = {}
        self.dov: list = []
        self.dov_mean: float = 0.0
        self.dov_std: float = 0.0
        self.his_dov_mean: list = []
        self.his_dov_std: list = []

        # 跨日信号执行
        self.last_day_signal = None
        self.current_date: Optional[date] = None

        # 从本地文件加载交易日历
        self.trading_days = self.load_local_trading_days(
            '/Users/YourName/Desktop/ReportReplication/local_trading_days_1924.csv'
        )

    def on_init(self):
        """初始化时自动调用，一般做数据预加载等。"""
        self.load_bar(days=100)
        self.write_log('策略初始化完成.')

    def on_start(self):
        """策略启动时调用。"""
        self.write_log('策略启动.')

    def on_stop(self):
        """策略停止时调用。"""
        self.write_log('策略停止.')

    def on_bar(self, bar: BarData):
        """
        每收到一根 Bar(假设是1分钟级别) 就会调用一次。
        在这里完成对日内 OI 拆分、计算 DOV、检查止盈止损等工作。
        """
        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        bar_date = bar.datetime.date()
        # 如果尚未记录今天的日期，就初始化一下
        if not self.current_date:
            self.current_date = bar_date

        # 如果日期变了，说明开盘跨日了，要执行跨日信号
        if bar_date != self.current_date:
            self.cross_day_ordering(bar)
            self.reset_daily_variables()
            self.current_date = bar_date

        # 第一根 Bar 的时候，先存储一下第一分钟的 OI / close
        if len(self.oi_adj_value) == 0:
            self.oi_adj_value.append(bar.open_interest)
            self.close_list.append(bar.close_price)

        # 从第二根 Bar 开始，才能算差分
        if self.last_bar:
            # 每分钟 OI 的增量
            oi_minu_change = bar.open_interest - self.last_bar.open_interest
            self.oi_minu_change.append(oi_minu_change)

            # 每分钟成交量的增量
            vol_min_change = bar.volume - self.last_bar.volume
            self.vol_minu_change.append(vol_min_change)

            # 当日的累加
            self.daily_volume += bar.volume
            self.oi_daily_change += oi_minu_change

            self.close_list.append(bar.close_price)
            self.volume_list.append(bar.volume)

            # 检查止盈止损
            self.check_exit_conditions(bar)

            # 如果是收盘前最后一分钟(示例判断 14:58)
            if self.last_bar.datetime.hour == 14 and self.last_bar.datetime.minute == 58:
                self.cancel_all()
                # 计算拆分后的 OI 序列
                self.calculate_oi_adjusted_sequence(bar)
                # 计算 DOV
                self.calculate_intraday_dov()
                # 生成当日信号，准备跨日使用
                self.decide_daily_signal(bar)

        self.last_bar = bar
        self.put_event()

    def calculate_oi_adjusted_sequence(self, bar: BarData):
        """
        计算当日每分钟的 OI 拆分(T+0 和 T+1)，再得到修正后的 OI 序列。
        最终会计算出当日的价量相关系数 PV，并存在 pv_dic 中。
        """
        vol_array = np.array(self.volume_list, dtype=float)
        if vol_array.sum() == 0:
            return

        # 用当天每分钟的成交量 / 当天总成交量 作为权重
        w_array = vol_array / vol_array.sum()
        # T+1 交易者的 oi 变化分配
        t1_array = w_array * self.oi_daily_change
        self.oi_change_t1 = list(t1_array)

        oi_minu_array = np.array(self.oi_minu_change, dtype=float)
        # T+0 交易者的 oi变化： = -(oi_minu_change - t1_array)
        t0_array = -1 * (oi_minu_array - t1_array)
        self.oi_change_t0 = list(t0_array)

        # 计算修正后的 OI 序列
        pre_oi = self.oi_adj_value[0]
        for i in range(len(vol_array)):
            current_oi = pre_oi + self.oi_change_t0[i] + self.oi_change_t1[i]
            self.oi_adj_value.append(current_oi)
            pre_oi = current_oi

        # 再基于收盘价序列 vs 修正后的 OI 序列计算相关系数 PV
        if len(self.close_list) > 1 and len(self.oi_adj_value) > 1:
            pv = np.corrcoef(self.close_list, self.oi_adj_value)[0, 1]
            self.pv_dic[bar.datetime.date()] = pv

    def calculate_intraday_dov(self):
        """
        计算当日的 DOV 序列，每分钟 (|∆oi| / ∆volume)，再得到其均值和标准差，用来判断是否反转。
        """
        minute_dov_list = []
        for oi_change, vol_change in zip(self.oi_minu_change, self.vol_minu_change):
            if vol_change != 0:
                minute_dov = abs(oi_change) / vol_change
            else:
                minute_dov = 0.0
            minute_dov_list.append(minute_dov)

        self.dov = minute_dov_list
        self.dov_mean = np.mean(self.dov)
        self.dov_std = np.std(self.dov)
        self.his_dov_mean.append(self.dov_mean)
        self.his_dov_std.append(self.dov_std)

    def decide_daily_signal(self, bar: BarData):
        """
        判断当日(收盘前)要给下一交易日发出的开仓信号：'long' or 'short' or None.
        并结合 DOV 的大小决定要不要反转该信号。
        """
        if bar.datetime.date() not in self.pv_dic:
            self.last_day_signal = None
            return

        # 首先根据 PV 的正负来判断信号
        pv = self.pv_dic[bar.datetime.date()]
        candidate_signal = 'long' if pv > 0 else 'short'

        # 如果有足够历史数据，则根据 DOV 判断是否要反转
        if (
            len(self.his_dov_mean) >= self.dov_mean_lookback and
            len(self.his_dov_std) >= self.dov_std_lookback
        ):
            # 如果 DOV 均值或标准差过高，说明机构交易者多 or 投资者结构波动大 => 可能要反转
            if self.compare_with_his_percentile_mean(
                self.his_dov_mean,
                self.dov_mean_lookback,
                self.dov_mean_percentile
            ) or self.compare_with_his_percentile_mean(
                self.his_dov_std,
                self.dov_std_lookback,
                self.dov_std_percentile
            ):
                # Flip the signal
                candidate_signal = 'long' if candidate_signal == 'short' else 'short'
        else:
            pass

        # 如果下一交易日不可交易，或者离得太久(长假)，则不开仓
        next_trading_day = self.get_next_trading_day(bar.datetime.date())
        if not next_trading_day:
            candidate_signal = None
        else:
            if self.days_between(bar.datetime.date(), next_trading_day) > 3:
                candidate_signal = None

        self.last_day_signal = candidate_signal

    def cross_day_ordering(self, bar: BarData):
        """
        在进入新交易日时，根据 last_day_signal 发出开仓/平仓指令。
        """
        if self.last_day_signal == 'long':
            if self.pos == 0:
                self.buy_price = bar.close_price
            elif self.pos < 0:
                # 如果之前是空头，就要先反手
                self.cover_price = bar.close_price
                self.buy_price = bar.close_price

        elif self.last_day_signal == 'short':
            if self.pos == 0:
                self.short_price = bar.close_price
            elif self.pos > 0:
                # 如果之前是多头，就要先反手
                self.sell_price = bar.close_price
                self.short_price = bar.close_price

        self.send_order_orderly()

    def reset_daily_variables(self):
        """
        清空当日的临时列表，方便明天重新累加。
        """
        self.daily_volume = 0
        self.oi_minu_change.clear()
        self.oi_daily_change = 0
        self.close_list.clear()
        self.volume_list.clear()
        self.vol_minu_change.clear()
        self.w_list.clear()
        self.oi_change_t1.clear()
        self.oi_change_t0.clear()
        self.oi_adj_value.clear()
        self.dov.clear()
        self.dov_mean = 0.0
        self.dov_std = 0.0

    def load_local_trading_days(self, file_path: str):
        """
        从指定文件路径读取交易日历(假设是csv)，并返回一个排序后的列表。
        文件需包含一列 trade_date 格式为YYYY-MM-DD。
        """
        df = pd.read_csv(file_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
        trading_days = sorted(df["trade_date"].tolist())
        return trading_days

    def is_trading_day(self, day: date) -> bool:
        return day in self.trading_days

    def get_next_trading_day(self, current_day: date) -> Optional[date]:
        future_days = [d for d in self.trading_days if d > current_day]
        if future_days:
            return future_days[0]
        return None

    def days_between(self, start: date, end: date) -> int:
        return (end - start).days

    def compare_with_his_percentile_mean(self, data_list, lookback, percentile):
        """
        判断最后一个值是否同时大于：
          1) data_list在回看 lookback 区间内的 percentile分位数
          2) 最近5个值(不包括最后1个)的均值
        若都大于, 则返回 True
        """
        if len(data_list) < lookback:
            return False

        last_value = data_list[-1]
        recent_5_values = data_list[-6:-1]
        if len(recent_5_values) == 0:
            return False

        recent_mean = np.mean(recent_5_values)
        data_list_percentile = data_list[-(lookback+1):-1]
        percentile_value = np.percentile(data_list_percentile, percentile)

        return (last_value > percentile_value) and (last_value > recent_mean)

    def send_order_orderly(self):
        """
        按序发送开仓/平仓指令，以免相互冲突。
        """
        if self.pos == 0:
            # 如果没有仓位，看看是否需要开仓
            if self.buy_price > 0:
                self.buy(price=self.buy_price, volume=self.fixed_size, stop=False)
                self.buy_price = 0
            elif self.short_price > 0:
                self.short(price=self.short_price, volume=self.fixed_size, stop=False)
                self.short_price = 0

        elif self.pos > 0:  # 有多头仓位
            if self.sell_price > 0:
                self.sell(price=self.sell_price, volume=abs(self.pos), stop=True)
                self.sell_price = 0

        elif self.pos < 0:  # 有空头仓位
            if self.cover_price > 0:
                self.cover(price=self.cover_price, volume=abs(self.pos), stop=True)
                self.cover_price = 0

        self.sync_data()

    def check_exit_conditions(self, bar: BarData):
        """
        检查日内止盈止损条件，如满足则立刻发出指令。
        """
        if self.pos > 0:
            # 多头止损
            if bar.close_price < self.entry_price * (1 - self.stop_loss_pct):
                self.sell(price=bar.close_price, volume=abs(self.pos), stop=True)
            # 多头止盈
            elif bar.close_price > self.entry_price * (1 + self.take_profit_pct):
                self.sell(price=bar.close_price, volume=abs(self.pos), stop=True)

        elif self.pos < 0:
            # 空头止损
            if bar.close_price > self.entry_price * (1 + self.stop_loss_pct):
                self.cover(price=bar.close_price, volume=abs(self.pos), stop=True)
            # 空头止盈
            elif bar.close_price < self.entry_price * (1 - self.take_profit_pct):
                self.cover(price=bar.close_price, volume=abs(self.pos), stop=True)

    def on_order(self, order: OrderData):
        """当限价单状态更新时调用。"""
        pass

    def on_stop_order(self, stop_order: StopOrder):
        """当本地下的停止单被触发时调用。"""
        pass

    def on_trade(self, trade: TradeData):
        """
        当委托成交时调用。若是开仓则记录开仓价；若是全部平仓，则检测是否需要继续下单。
        """
        if trade.offset != Offset.CLOSE:
            # 如果是开仓，记录开仓价
            self.entry_price = trade.price

        # 如果这次成交是平仓，并且仓位已经回到0，就看看是否需要下一步发单
        if trade.offset == Offset.CLOSE and self.pos == 0:
            self.send_order_orderly()