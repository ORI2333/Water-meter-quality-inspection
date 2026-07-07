# FPGA DDR3 + AXI4 帧缓冲控制器实现指南

## 目录
1. [系统架构](#1-系统架构)
2. [DDR3 硬件与 MIG 配置](#2-ddr3-硬件与-mig-配置)
3. [AXI4 协议要点](#3-axi4-协议要点)
4. [axi4_ctrl 模块详解](#4-axi4_ctrl-模块详解)
5. [时钟域与帧缓冲管理](#5-时钟域与帧缓冲管理)
6. [关键 Bug 与修复](#6-关键-bug-与修复)
7. [调试方法](#7-调试方法)

---

## 1. 系统架构

本设计实现了一条完整的图像数据通路：摄像头/测试图案 → DDR3 帧缓冲 → HDMI 显示。

```
                          ┌─────────────┐
  SC1336 Sensor ─────────→│             │
  (DVP 8-bit)             │  axi4_ctrl  │     ┌──────────┐     ┌─────────┐
                          │             │────→│  DDR3_0  │────→│         │
  Test Pattern @24MHz ───→│  W0_FIFO    │     │  (MIG)   │     │  HDMI   │
                          │  256b×?     │     │  MT41K.. │     │ 720p@60 │
                          │             │←────│  667MHz  │────→│         │
                          │  R0_FIFO    │     └──────────┘     └─────────┘
                          │  256b×?     │
                          └─────────────┘
                            ↑        ↓
                         clk_cmos  clk_video
                         (24MHz)   (74.25MHz)
```

**核心设计原则：写读时钟域分离。**

- 写侧用 `clk_cmos`（24MHz），独立于显示时序
- 读侧用 `clk_video`（74.25MHz），与 LCD 扫描同步
- 4 个帧缓冲区避免读写冲突

---

## 2. DDR3 硬件与 MIG 配置

### 2.1 硬件规格

| 参数 | 值 |
|------|-----|
| FPGA | XC7K70T-FBG676 (-3) |
| DDR3 芯片 | MT41K128M16XX-15E × 2pcs |
| 数据位宽 | 32-bit（两片 16-bit 拼接） |
| DDR 频率 | 666.67MHz (1333 MT/s) |
| 带宽 | 32bit × 1333MT/s = 5.3 GB/s |

### 2.2 MIG IP 配置

MIG（Memory Interface Generator）是 Xilinx 提供的 DDR3 控制器 IP 核。关键参数：

```
Design Clock Frequency         : 666.67 MHz (1500 ps)
PHY to Controller Clock Ratio  : 4:1
Input Clock Period             : 3999 ps   (~250 MHz)
CLKFBOUT_MULT (PLL)            : 16
DIVCLK_DIVIDE (PLL)            : 3
Memory Part                    : MT41K128M16XX-15E
Data Width                     : 32
AXI Data Width                 : 256
ID Width                       : 4
Address Mapping                : BANK_ROW_COLUMN
SysResetPolarity               : ACTIVE LOW   ← 关键！
```

**理解 PHY to Controller Ratio (4:1)**：
- DDR PHY 跑 666.67MHz
- UI Clock = 666.67 / 4 = 166.67MHz
- 每个 UI 时钟周期传输 4 个 DDR 数据拍（256-bit）

### 2.3 MIG 用户接口信号

MIG 以 AXI4 Slave 接口对外暴露，信号宽度如下：

```verilog
// 时钟与复位
input         sys_clk_i,        // 系统时钟 250MHz
input         clk_ref_i,        // IDELAYCTRL 参考钟 200MHz（必须！）
input         sys_rst,          // 系统复位（ACTIVE LOW）
output        ui_clk,           // 用户时钟 166.67MHz
output        ui_clk_sync_rst,  // 用户复位（高有效）
output        init_calib_complete, // 校准完成标志

// AXI Write Address
input  [3:0]  s_axi_awid,
input  [28:0] s_axi_awaddr,
input  [7:0]  s_axi_awlen,
input  [2:0]  s_axi_awsize,     // 2^5=32 Bytes = 256-bit
// ...

// AXI Write Data
input  [255:0] s_axi_wdata,
input  [31:0]  s_axi_wstrb,
// ...

// AXI Read Data
output [255:0] s_axi_rdata,
// ...
```

> **关键教训**：`clk_ref_i` 必须接 200MHz ±10MHz。这是 IDELAYCTRL 硬核的参考钟，不是可选的。我们的设计从 `sys_clk_ddr_ref` PLL 同时输出 250MHz 和 200MHz。

---

## 3. AXI4 协议要点

### 3.1 五个独立通道与 VALID/READY 握手

AXI4 协议定义了 5 个独立的单向通道，每个通道都使用相同的 VALID/READY 握手机制：

```
                    Master                         Slave
Write Address (AW): ──── awvalid/awaddr ────→  (awready)
Write Data    (W):  ──── wvalid/wdata ──────→  (wready)
Write Response (B):  ←── bvalid/bresp ───────  (bready)
Read Address  (AR):  ──── arvalid/araddr ────→  (arready)
Read Data     (R):   ←── rvalid/rdata ───────  (rready)
```

**VALID/READY 握手规则**：
- 发送方（Master）准备好数据后拉高 VALID，保持直到握手完成
- 接收方（Slave）准备好接收时拉高 READY
- **当 VALID && READY 同时为高的时钟沿，一次传输完成**
- VALID 不能等待 READY——发送方必须先拉高 VALID
- READY 可以在 VALID 之前拉高（Slave 提前表示可以接收）

```
clk     ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐
        ┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─
awvalid ────────┐           ┌─────────
awready ────────────┐   ┌───────────
                    │   │
                    ├───┤ ← 握手！传输在这一拍发生
wdata   ────────[ D0 ][ D1 ][ D2 ]...
```

### 3.2 AXI4 Burst 传输详解

本设计使用 INCR（递增）Burst 类型。一次 Burst 传输连续 16 拍数据（`awlen=15`）。

**写 Burst 时序**：

```
         ┌─ AW 握手 ─┐  ┌──────── W 通道 16 拍 ────────────┐ ┌─ B 响应 ─┐
awvalid  ──┐         ┌──
awready  ────┐     ┌────
awaddr   ────[ADDR]─────────────────────────────────────────────
wvalid   ───────────┐                                     ┌────
wready   ─────────────┐                               ┌──────
wdata    ─────────────[D0][D1][D2]...[D14][D15]─────────────
wlast    ───────────────────────────────────────┐       ┌─────
bvalid   ───────────────────────────────────────────┐ ┌──────
bready   ─────────────────────────────────────────────┐┌─────
```

关键点：
- AW 通道先于 W 通道发起（或同时）
- W 通道 16 拍数据连续发送
- 最后一拍（第 16 拍）`wlast=1`
- Burst 完成后，Slave 通过 B 通道回复写响应

**读 Burst 时序**：

```
         ┌─ AR 握手 ─┐  ┌──────── R 通道 16 拍 ────────────┐
arvalid  ──┐         ┌──
arready  ────┐     ┌────
araddr   ────[ADDR]─────────────────────────────────────────────
rvalid   ─────────────────┐                               ┌────
rready   ───────────────────┐                           ┌──────
rdata    ───────────────────[D0][D1][D2]...[D14][D15]──────
rlast    ───────────────────────────────────────┐       ┌─────
```

### 3.3 AXI4 地址计算与 DDR 内存布局

#### 3.3.1 地址拼接

DDR 地址空间使用 **{帧索引, 帧内偏移}** 的拼接方式。本设计中：

```
32-bit 地址:
┌──────────┬──────┬──────────────────────────┐
│ 未使用    │ 帧索引 │  帧内偏移 (22 bits = 4MB)  │
│ [31:24]  │ [23:22]│  [21:0]                    │
└──────────┴──────┴──────────────────────────┘
```

传递给 MIG 时截取低 29 位 `[28:0]`。

```verilog
// awaddr 计算——每行代码对应硬件上的一个加法器
assign axi4_awaddr = BASE_ADDR + {rc_wframe_index, rc_w_ptr};
//                   = 0x00000000 + {2'dN, 22'dOffset}
//                   = N × 4MB + Offset

// araddr 计算——与 awaddr 对称
assign axi4_araddr = BASE_ADDR + {rc_rframe_index, araddr};
//                   = 0x00000000 + {2'dN, 22'dOffset}
```

#### 3.3.2 帧内偏移 (rc_w_ptr / araddr)

每写完一个 512 字节的 Burst，指针增加 `ADDR_INC = 16 × 32 = 512`：

```
Burst 0:  rc_w_ptr = 0        → awaddr = 0x000000
Burst 1:  rc_w_ptr = 512      → awaddr = 0x000200
Burst 2:  rc_w_ptr = 1024     → awaddr = 0x000400
...
Burst 1799: rc_w_ptr = 1799×512 = 921088  → awaddr = 0x0E0E00
```

一帧共 921600 字节（1280×720×1 byte），1799 个完整的 512 字节 Burst  = 1799×512 = 921088 字节，最后一 Burst 不足 512 字节（仅 512 字节中的前 512 字节有效）。实际 1800 个 Burst 共 1800×512 = 921600 字节，恰好一帧。

#### 3.3.3 地址递增参数计算

```verilog
parameter BURST_LEN = 16;                    // 每 Burst 16 拍
parameter STRB_LEN  = DATA_LEN / 8;          // 256/8 = 32 字节/拍
parameter ADDR_INC  = BURST_LEN * STRB_LEN;  // 16 × 32 = 512 字节/Burst
parameter BUF_SIZE  = 22;                    // 2^22 = 4,194,304 字节/缓冲
parameter BASE_ADDR = 32'h0000_0000;         // DDR 起始地址 = 0
```

4 个帧缓冲区的地址分布：

```
Buffer 0: 0x00000000 ~ 0x003FFFFF  (0 ~ 4MB)
Buffer 1: 0x00400000 ~ 0x007FFFFF  (4MB ~ 8MB)
Buffer 2: 0x00800000 ~ 0x00BFFFFF  (8MB ~ 12MB)
Buffer 3: 0x00C00000 ~ 0x00FFFFFF  (12MB ~ 16MB)
```

### 3.4 awvalid 与 awready 的精确保序

```verilog
// awvalid 在 IDLE 态发起，在收到 awready 后清除
if (axi4_awvalid && axi4_awready)
    axi4_awvalid <= 1'b0;   // 握手完成，拉低 valid

// 在 WS_W_IDLE 态，FIFO 有数据时重新拉高
WS_W_IDLE: begin
    if (~w_wfifo_pempty || ...) begin
        axi4_awvalid <= 1'b1;   // 发起新的写地址请求
    end
end
```

关键约束：**同一时刻只能有一个未完成的写地址**（awvalid 拉高后，必须等 awready 握手完成才能再拉高）。

### 3.5 awsize 必须匹配数据位宽

`awsize` 编码表：

| awsize | 字节/拍 | 适用数据位宽 |
|--------|---------|------------|
| 0 | 1 | 8-bit |
| 1 | 2 | 16-bit |
| 2 | 4 | 32-bit |
| 3 | 8 | 64-bit |
| 4 | 16 | 128-bit |
| **5** | **32** | **256-bit** ← 本设计 |
| 6 | 64 | 512-bit |
| 7 | 128 | 1024-bit |

> **致命教训**：曾经 awsize=4（16 字节/拍），但数据位宽是 256-bit=32 字节。MIG 每拍只存取低 128-bit，高位数据永久丢失→雪花屏。

---

## 4. axi4_ctrl 模块详解

### 4.1 模块参数

```verilog
module axi4_ctrl #(
    parameter ID_LEN       = 8,          // AXI ID 宽度
    parameter ADDR_LEN     = 32,         // 地址宽度
    parameter DATA_LEN     = 256,        // 数据位宽 (bit)
    parameter DATA_SIZE    = 5,          // 2^5=32 字节/拍
    parameter BURST_LEN    = 16,         // Burst 拍数
    parameter STRB_LEN     = DATA_LEN/8, // 写选通宽度
    parameter ADDR_INC     = BURST_LEN * STRB_LEN, // 512 字节
    parameter W_WIDTH      = 8,          // 写数据输入位宽
    parameter R_WIDTH      = 8,          // 读数据输出位宽
    parameter BUF_SIZE     = 22,         // 2^22 = 4MB 缓冲区
    parameter RD_END_ADDR  = 1280*720,   // 一帧像素数
    parameter BASE_ADDR    = 32'h0000_0000
)(
    input  wire        axi4_clk,         // AXI 时钟 (ui_clk 166.67MHz)
    input  wire        axi4_rst_n,       // AXI 复位 (低有效)

    // AXI4 五个通道 (略，详见源码)
    // ...

    // 帧接口
    input  wire        wframe_pclk,      // 写像素时钟 (24MHz)
    input  wire        wframe_vsync,     // 写帧同步
    input  wire        wframe_data_en,   // 写数据有效
    input  wire [7:0]  wframe_data,      // 写数据

    input  wire        rframe_pclk,      // 读像素时钟 (74.25MHz)
    input  wire        rframe_vsync,     // 读帧同步
    input  wire        rframe_data_en,   // 读数据请求
    output wire [7:0]  rframe_data       // 读数据输出
);
```

### 4.2 写数据通路

数据从窄位宽（8-bit）拼成宽位宽（256-bit）再写 DDR：

```verilog
// 移位寄存器：每次移入 W_WIDTH=8 位
reg  [DATA_LEN-1:0] r_wfifo_wdata;
wire [DATA_LEN-1:0] w_wfifo_wdata;
assign w_wfifo_wdata = {wframe_data, r_wfifo_wdata[DATA_LEN-1:W_WIDTH]};

always @(posedge wframe_pclk) begin
    if (!r_w_rst_n) begin
        r_wfifo_wdata <= {DATA_LEN{1'b0}};   // 帧结束时清空
    end else if (wframe_data_en) begin
        r_wfifo_wdata <= w_wfifo_wdata;       // 每有效像素移入 8 位
    end
end
```

**工作原理**：
1. 每来一个像素（`wframe_data_en=1`），8-bit 数据从右侧移入移位寄存器
2. 累计 32 个像素（32×8=256 bit）后，写入 W0_FIFO
3. 写状态机从 W0_FIFO 读出 256-bit 字，通过 AXI Burst 写入 DDR

**帧结束处理**：
```verilog
WS_W_EOF: begin
    r_weof_pending <= 1'b0;      // 清除 EOF 标志
    rc_w_ptr       <= 1'b0;      // 写指针归零（新帧从头开始）
    state_write    <= WS_W_IDLE;  // 回到空闲
end
```

### 4.3 写状态机

```verilog
localparam WS_W_IDLE  = 2'b00;   // 空闲：等待 FIFO 数据
localparam WS_W_WDATA = 2'b01;   // 写数据：发送 Burst
localparam WS_W_WINC  = 2'b10;   // 指针递增：Burst 完成
localparam WS_W_EOF   = 2'b11;   // 帧结束：复位指针

case (state_write)
    WS_W_IDLE: begin
        if (~w_wfifo_pempty || ...) begin
            axi4_awvalid <= 1'b1;              // 发起写请求
            axi4_wvalid  <= 1'b1;
            state_write  <= WS_W_WDATA;
        end else if (r_weof_pending) begin
            r_w_rst_n <= 1'b0;                 // 帧结束，复位 FIFO
            r_wframe_inc <= 1'b1;              // 通知帧索引管理
            state_write <= WS_W_EOF;
        end
    end

    WS_W_WDATA: begin
        rc_burst <= rc_burst + axi4_wready;
        if (axi4_wlast && axi4_wready) begin   // Burst 最后一拍
            axi4_wvalid <= 1'b0;
            state_write <= WS_W_WINC;
        end
    end

    WS_W_WINC: begin
        rc_w_ptr <= rc_w_ptr + ADDR_INC;       // 指针增加 512 字节
        state_write <= WS_W_IDLE;
    end

    WS_W_EOF: begin
        r_weof_pending <= 1'b0;
        rc_w_ptr <= 1'b0;
        state_write <= WS_W_IDLE;
    end
endcase
```

> **致命教训**：曾经的 case 标签中 `WS_W_WINC` 被错误标记为 `WS_W_EOF`（重复标签），导致状态 `2'b10` 无匹配分支 → 写完第一个 Burst 后永久死锁。修复方法是确保 4 个编码各有唯一 case 标签。

### 4.4 读数据通路

从 DDR 读出 256-bit 字，拆成 8-bit 像素输出：

```verilog
// 计数器：每读取一个像素 +1，0→31 循环
localparam RFIFO_CNT_SIZE = (R_WIDTH == 8) ? 5 : ...;
reg [RFIFO_CNT_SIZE-1:0] rc_rfifo_rd;

always @(posedge rframe_pclk or posedge r_rfifo_rst_rclk) begin
    if (r_rfifo_rst_rclk) begin
        rc_rfifo_rd <= 'd0;
    end else if (rframe_data_en) begin
        rc_rfifo_rd <= rc_rfifo_rd + 1'b1;
    end
end

// 在 rc_rfifo_rd==0 时从 FIFO 加载新 256-bit 字
assign w_rframe_data_en_gen = rframe_data_en && (rc_rfifo_rd == 0)
                              && !w_rfifo_empty;

// 数据移位输出
reg [255:0] r_rframe_data_gen;
always @(posedge rframe_pclk) begin
    if (w_rframe_data_en_gen) begin
        r_rframe_data_gen <= w_rframe_data_gen;   // 加载新字
    end else if (rframe_data_en) begin
        r_rframe_data_gen <= r_rframe_data_gen >> R_WIDTH; // 右移 8 位
    end
end
assign rframe_data = w_rfifo_empty ? 8'h00
                    : r_rframe_data_gen[R_WIDTH-1:0];
```

**工作原理**：
1. R0_FIFO 写侧在 `axi4_clk` 域接收 AXI 读数据
2. R0_FIFO 读侧在 `rframe_pclk` 域按需输出
3. 每 32 个像素触发一次 FIFO 读（`rc_rfifo_rd == 0`）
4. 加载 256-bit 字后，每次右移 8 位输出一个像素

#### 4.2.1 移位寄存器打包详解（时序图）

每来一个像素（`wframe_data_en=1`），8-bit 数据从右侧移入 256-bit 移位寄存器：

```
PCLK     ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ... ┌─┐ ┌─┐
wframe_data_en ────┐     ┌───┐     ┌─── ... ──┐
                   │     │   │     │           │
Pixel #:           P0    P1  P2    P3  ...     P31

r_wfifo_wdata[7:0]     ← P0
r_wfifo_wdata[15:8]    ← (shifted)
r_wfifo_wdata[23:16]   ← (shifted)
...
经过 32 个 data_en 后:
r_wfifo_wdata[7:0]   = P31 (最新)
r_wfifo_wdata[15:8]  = P30
...
r_wfifo_wdata[255:248] = P0 (最早)

此时 rc_wfifo_we = 5'b11111 = 31, (&rc_wfifo_we) = 1
→ W0_FIFO.wr_en = 1 → w_wfifo_wdata 被写入 FIFO
```

**PCLK 速率计算**（clk_cmos @24MHz）：
- 每秒采样 24M 像素
- 每 32 像素填满一个 256-bit 字 → 24M/32 = 750K 字/秒
- FIFO 写入带宽 = 750K × 32 bytes = 24 MB/s
- AXI 写带宽 = 166.67M × 32 bytes（每拍）× 16 拍/16 拍 = 5.3 GB/s
- 写不溢出：5.3 GB/s >> 24 MB/s

#### 4.2.2 帧结束时的移位寄存器清理

这是一个关键的边缘情况处理：

```
帧尾最后一拍数据（P_last）到达后:
  r_wfifo_wdata 中可能还残留 0~31 个像素未写入 FIFO

如果不清空:
  → 下一帧的前 31 个像素会和残留数据混合 → 每帧第一条水平条纹的 31 像素错乱 → "重影/撕裂"

修复：r_w_rst_n = 0 时清空整个移位寄存器
  r_wfifo_wdata <= {DATA_LEN{1'b0}};   // 256 位全零
```

> **致命教训**：R0_FIFO 的 `din` 曾错误连接到 FIFO 自己的 `dout`→自循环永远为空。`wr_clk` 曾设为 `rframe_pclk` 而非 `axi4_clk`。R0_FIFO 复位逻辑曾有双取反导致永远不复位的 bug。

#### 4.3.1 写状态机完整时序

```
          ┌──── IDLE ────┐ ┌──── WDATA ────┐ ┌─WINC┐ ┌─IDLE─┐
state:   2'b00            2'b01              2'b10   2'b00
awvalid: ──────┐         ┌──────────────────────
awready: ────────┐     ┌────────────────────────
wvalid:  ──────┐         ┌────────────────┐
wready:  ────┐             ┌────────────┐
wdata:   ────[D0][D1]...[D14][D15]───────
wlast:   ───────────────────────┐       ┌───
rw_ptr:  [   N    ]             │       [N+512]
rc_burst: 0 → 1 → 2 → ... → 15 │       0
                                │
                           Burst 完成，指针+512
```

**状态转移条件**：
- IDLE → WDATA：FIFO 非空（`~w_wfifo_pempty`）或帧结束+FIFO 有残留数据
- WDATA → WINC：最后一拍握手完成（`axi4_wlast && axi4_wready`）
- WINC → IDLE：无条件，单周期
- IDLE → EOF：帧结束标志 & FIFO 全空
- EOF → IDLE：单周期，完成指针归零和帧索引递增

### 4.4 读数据通路（完整版）

#### 4.4.1 R0_FIFO 跨时钟域架构

```
  axi4_clk 域 (166.67MHz)              rframe_pclk 域 (74.25MHz)
  ┌─────────────────────┐              ┌──────────────────────┐
  │ axi4_rdata[255:0]   │              │ rframe_data[7:0]     │
  │        ↓            │              │        ↑             │
  │ rfifo_wenb (1-cycle)│   R0_FIFO    │ r_rframe_data_gen    │
  │ rfifo_wdata[255:0]  │════256b═════→│   >> 8 each cycle    │
  │        ↓            │  Async FIFO  │        ↑             │
  │   (FIFO write)      │              │ w_rframe_data_gen    │
  └─────────────────────┘              │   (FIFO output)      │
                                       │        ↑             │
                                       │ rc_rfifo_rd counter  │
                                       │  0→31→0→31...        │
                                       └──────────────────────┘
```

#### 4.4.2 解包计数器与 FIFO 读时序

每 32 个像素（`rc_rfifo_rd == 0`）触发一次 FIFO 读：

```
Pixel #:    0  1  2 ... 30 31  0  1 ...
rd_count:   0  1  2 ... 30 31  0  1 ...
rd_en:     ┌┐                       ┌┐
           └┘                       └┘
           加载新 256b 字            加载下一个字
           
rdata:   [P0][P1][P2]...[P30][P31][P0][P1]...
          ↑                       ↑
      r_rframe_data_gen[7:0]   r_rframe_data_gen[7:0]
      after load               after next load
```

**空 FIFO 保护**：若 FIFO 空（`w_rfifo_empty=1`），`w_rframe_data_en_gen=0`→不加载→输出 0→显示黑像素（瞬时闪过，比撕裂/花屏好）。

#### 4.4.3 R0_FIFO 写侧（AXI 时钟域）

```verilog
// 当 AXI 读数据到达时，写入 R0_FIFO
always @(posedge axi4_clk) begin
    if (axi4_rvalid && axi4_rready) begin
        rfifo_wenb  <= 1'b1;        // 单周期写使能
    end else begin
        rfifo_wenb  <= 1'b0;
    end
end

always @(posedge axi4_clk) begin
    rfifo_wdata <= axi4_rdata;      // 延迟一拍对齐写使能
end
```

关键时序：`rfifo_wenb` 和 `rfifo_wdata` 是同一拍产生的——data 在 cycle N 采样，wen 在 cycle N 拉高，data 在 cycle N+1 呈现 → 此时 wen=1 → FIFO 写入正确数据。

#### 4.4.4 读状态机详细流程

```verilog
// 读地址递增管理
always @(posedge axi4_clk or posedge r_rfifo_rst) begin
    if (r_rfifo_rst)
        araddr <= 0;                       // 帧复位：归零
    else if (axi4_arvalid && axi4_arready)
        araddr <= araddr + ADDR_INC;        // Burst 完成：+512
end

// 读完成跟踪
always @(posedge axi4_clk or posedge r_rfifo_rst) begin
    if (r_rfifo_rst) begin
        rd_state    <= S_READ_IDLE;
        r_rd_pend   <= 0;
        axi4_arvalid <= 0;
    end else begin
        // arvalid 在握手后清除
        if (axi4_arready && axi4_arvalid)
            axi4_arvalid <= 0;
        
        // r_rd_pend 在最后一拍数据到达时清除
        if (axi4_rvalid && axi4_rlast)
            r_rd_pend <= 0;
        
        case (rd_state)
            S_READ_IDLE:
                if (ready) begin           // FIFO 稳定 && 地址未超 && FIFO 未满
                    rd_state    <= S_READ_ADDR;
                end
            S_READ_ADDR: begin
                axi4_arvalid <= 1;         // 发起读请求
                r_rd_pend    <= 1;
                rd_state     <= S_READ_DATA;
            end
            S_READ_DATA:
                if (!axi4_arvalid && !r_rd_pend) // Burst 全部完成
                    rd_state <= S_READ_IDLE;
        endcase
    end
end
```

### 4.5 W0_FIFO 与 R0_FIFO 的完整连接

```verilog
// ========== W0_FIFO：写侧 FIFO ==========
W0_FIFO_256 u_W0_FIFO_256 (
    .rst        (~r_w_rst_n),          // 帧结束时复位（高有效）
    .wr_clk     (wframe_pclk),         // 写时钟 = 像素时钟 (24MHz)
    .wr_en      (wframe_data_en && (&rc_wfifo_we)), // 32 像素写一次
    .din        (w_wfifo_wdata),       // 拼接后的 256-bit 数据
    .rd_clk     (axi4_clk),            // 读时钟 = AXI 时钟 (166.67MHz)
    .rd_en      (w_wfifo_ren),         // axi4_wvalid && axi4_wready
    .dout       (w_wfifo_rdata),       // 输出到 AXI 写数据总线
    .empty      (w_wfifo_empty),
    .prog_empty (w_wfifo_pempty)       // 几乎空：触发 Burst 的条件
);

// ========== R0_FIFO：读侧 FIFO ==========
R0_FIFO_256 u_R0_FIFO_256 (
    .rst        (w_rfifo_rst),         // 帧边界复位（高有效）
    .wr_clk     (axi4_clk),            // 写时钟 = AXI 时钟 ← 必须是 axi4_clk!
    .wr_en      (rfifo_wenb),          // axi4_rvalid && axi4_rready
    .din        (rfifo_wdata),         // AXI 读回的数据 ← 必须接 rfifo_wdata!
    .rd_clk     (rframe_pclk),         // 读时钟 = 像素时钟 (74.25MHz)
    .rd_en      (w_rframe_data_en_gen),// 每 32 像素读一次
    .dout       (w_rframe_data_gen),   // 输出到解包移位寄存器
    .empty      (w_rfifo_empty),
    .prog_empty (w_rfifo_aempty)
);
```

> **W0_FIFO 和 R0_FIFO 都是异步 FIFO**——写读两侧在不同时钟域。这要求 FIFO IP 配置为独立时钟模式（Independent Clocks），且数据位宽 256-bit，深度通常 512 或 1024。

```
S_READ_IDLE → S_READ_ADDR → S_READ_DATA → S_READ_IDLE
     ↑                           │
     └───────────────────────────┘
```

```verilog
case (rd_state)
    S_READ_IDLE: begin
        // 条件：FIFO 稳定 && 地址未到帧尾 && FIFO 未满
        if (rfifo_wr_rst_busy_dly[15]
            && (araddr < RD_END_ADDR)
            && (~rfifo_wfull))
            rd_state <= S_READ_ADDR;
    end
    S_READ_ADDR: begin
        axi4_arvalid <= 1;          // 发起读请求
        r_rd_pend    <= 1;          // 标记有待处理读
        rd_state     <= S_READ_DATA;
    end
    S_READ_DATA: begin
        if ((~axi4_arvalid) && (~r_rd_pend))
            rd_state <= S_READ_IDLE; // Burst 完成
    end
endcase
```

---

## 5. 时钟域与帧缓冲管理

### 5.1 时钟域划分

| 时钟 | 频率 | 用途 |
|------|------|------|
| `clk_50m` | 50MHz | 板上晶振，PLL 输入 |
| `clk_ctrl` | 100MHz | I2C、UART 控制域 |
| `clk_cmos` | 24MHz | 传感器时钟、DDR 写域 |
| `clk_video` | 74.25MHz | LCD 像素时钟、DDR 读域 |
| `clk_ddr_ref` | 250MHz | MIG 系统时钟 |
| `clk_ddr_ref_200m` | 200MHz | MIG IDELAYCTRL 参考钟 |
| `ui_clk` | 166.67MHz | MIG 用户接口、AXI 总线域 |

### 5.2 帧缓冲管理

使用 4 个缓冲区实现读写并行，避免冲突：

```verilog
// 缓冲区分配：每个 4MB，4 个共 16MB
// rc_wframe_index: 写指针 (0→1→2→3→0→...)
// rc_rframe_index: 读指针 (初始 2，远离写指针)

// 写完成后
rc_wframe_index <= w_wframe_index_next;       // 移到下一个安全位置
r_wframe_index_last <= rc_wframe_index;       // 记录刚写完的缓冲

// 读开始时（仅在写完成至少一帧后）
if (wframe_done)
    rc_rframe_index <= r_wframe_index_last;   // 读取刚写完的缓冲
```

**wframe_done 保护机制**：防止读在写完成前切到未写完的缓冲。

```
初始: 写→buf0, 读→buf2(空)
帧0完成: wframe_done=1, 写→buf1, 读→buf0(有效!)
帧1完成: 写→buf2, 读→buf1
帧2完成: 写→buf3, 读→buf2
帧3完成: 写→buf0, 读→buf3  (循环)
```

### 5.3 写读分离——根治撕裂

**核心设计决策**：写和读使用不同时钟域。

```
写侧: clk_cmos (24MHz)          读侧: clk_video (74.25MHz)
  XOR帧生成器 (24MHz, 独立时序)     LCD 驱动时序
       ↓                               ↓
  W0_FIFO (async)                 R0_FIFO (async)
       ↓                               ↓
  axi4_clk (166.67MHz) ←── DDR3 ──→ axi4_clk
```

因为写和读用完全不同的时钟、不同的帧生成器、不同的 vsync 信号，它们永远不可能在同一时刻操作同一个缓冲。撕裂从根本上被消除了。

> 曾尝试让写读共用 `clk_video` 和 `lcd_vs`——反复出现撕裂，因为写读在同一时刻竞争访问相邻缓冲。

---

## 6. 关键 Bug 与修复

### 6.1 `axi4_awaddr` 未驱动（致命）

**症状**：DDR 写响应正常（`bvalid` 脉冲），但读回全是随机垃圾。

**根因**：
```verilog
// 错误代码
assign awaddr = BASE_ADDR + {rc_wframe_index, rc_w_ptr}; // awaddr 未声明！隐式 1-bit wire
// axi4_awaddr 端口从未被赋值！→ 悬空
```

**修复**：
```verilog
assign axi4_awaddr = BASE_ADDR + {rc_wframe_index, rc_w_ptr};
```

> 这是雪花屏的根本原因。写地址永远为 0——所有数据写到地址 0，但读地址正确。读读到的是随机的 DDR 残留数据。

### 6.2 `awsize` 错误（致命）

**症状**：每个 AXI 字有一半数据丢失。

**根因**：`DATA_SIZE = 4` → `awsize = 4` → 2^4 = 16 字节/拍。但数据位宽是 256-bit = 32 字节。

**修复**：`DATA_SIZE = 5` → `awsize = 5` → 2^5 = 32 字节/拍。

### 6.3 MIG `sys_rst` 极性错误（致命）

**症状**：DDR 永远不校准（`ui_clk_sync_rst` 永久为 1）。

**根因**：MIG 配置 `SysResetPolarity = ACTIVE LOW`，但代码生成了 ACTIVE HIGH 复位。

**修复**：`assign mig_sys_rst = sys_rst_n;` ——直接用低有效系统复位。

### 6.4 R0_FIFO `din` 自循环（致命）

```verilog
// 错误
.din (w_rframe_data_gen),    // w_rframe_data_gen 是 FIFO 的 dout！
.dout(w_rframe_data_gen),    // 同一个 wire！

// 正确
.din (rfifo_wdata),          // AXI 读数据
.dout(w_rframe_data_gen),    // FIFO 输出
.wr_clk(axi4_clk),           // 写时钟 = AXI 时钟（不是 rframe_pclk！）
```

### 6.5 R0_FIFO 复位双取反（致命）

```verilog
// 错误
wire w_rfifo_rst_n = (~axi4_rst_n) || (~rfifo_rst_n); // 命名为 _n 但存 active-high
R0_FIFO .rst(~w_rfifo_rst_n);  // 又取反一次 = 双取反 = 和原意相反

// 正确
wire w_rfifo_rst = (~axi4_rst_n) || (~rfifo_rst_n); // active-high
R0_FIFO .rst(w_rfifo_rst);
```

### 6.6 写状态机 case 标签错位（致命）

4 状态中 `WS_W_WINC (2'b10)` 的 case 标签被错误写成了重复的 `WS_W_EOF` → 无匹配 → default → 死锁。

### 6.7 顶层参数覆盖（隐蔽）

`axi4_ctrl.sv` 默认值 `DATA_SIZE=5`，但 `sc1336_hdmi.sv` 例化时写 `.DATA_SIZE(4)`——顶层覆盖了模块默认值。

### 6.8 读帧索引起始值

`rc_rframe_index` 初始为 `2'd0`（和写指针相同）→ 第一帧读写撞 buffer。
修复为 `2'd2`（远离写指针 2 个 buffer）。

---

## 7. 调试方法

### 7.1 分步验证策略

```
Step 1: LCD 测试图案 (bypass DDR)
  → 验证 LCD 驱动、时钟、复位正常

Step 2: 常量写入 DDR (0xA5)
  → 纯灰屏 = DDR 通路正常

Step 3: 静态 XOR 图案
  → 确认地址计算、数据完整性

Step 4: 动态图案 (lcd_display_test)
  → 验证帧切换、多缓冲

Step 5: 传感器输入
  → 真实数据通路
```

### 7.2 UART 快速切换

4 条串口命令（115200 bps）在一秒内完成 A/B 对比：

| 命令 | 数据路径 | 验证内容 |
|------|---------|---------|
| `L` | 图案→LCD | LCD 完好？ |
| `G` | 0xA5→DDR→LCD | DDR 读写基础？ |
| `X` | XOR→DDR→LCD | 地址/数据完整？ |
| `D` | 动画→DDR→LCD | 帧缓冲/多帧？ |

### 7.3 ILA 关键观测点

```
Probe0: state_write 看写状态机是否在跑
Probe1: s_axi_rdata vs w_lcd_data 对比→定位故障在 DDR 侧还是 FIFO 侧
Probe2: ddr_init_done, rc_wframe_index, rc_rframe_index
```

### 7.4 常见故障模式

| 现象 | 可能原因 | 检查 |
|------|---------|------|
| 全黑 | PLL 未锁 / 一直复位 | `sys_rst_n`, `ddr_init_done` |
| 雪花 | 读写地址不匹配 | `awaddr` 是否正确驱动 |
| 撕裂 | 读写撞缓冲 / 同钟 | 时钟域是否分离 |
| 第一列异常 | 解包计数器偏移 | `rc_rfifo_rd` 时序 |
| 图案不切换 | vsync 未对齐 | 切换逻辑 vsync 门控 |

---

## 参考资料

- Xilinx UG586: 7 Series FPGAs Memory Interface Solutions
- ARM IHI 0022: AMBA AXI4 Protocol Specification
- MT41K128M16XX-15E Datasheet (Micron)
- 项目源码: `sc1336_hdmi/rtl/axi4_ctrl.sv`
