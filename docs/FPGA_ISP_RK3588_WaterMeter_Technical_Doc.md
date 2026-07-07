# FPGA ISP 涓? RK3588 姘磋〃鎸囬拡妫?娴嬫妧鏈枃妗?

鐗堟湰锛歏1.2

鏃ユ湡锛?2026-07-07

閫傜敤宸ョ▼锛?

- FPGA 宸ョ▼锛歚fpga/sc1336_hdmi_isp`
- RK3588 浠ｇ爜鐩綍锛歚/home/demo/water_meter/code`
- RK3588 妯″瀷鐩綍锛歚/home/demo/water_meter/module`

## 1. 绯荤粺鐩爣

鏈郴缁熺敤浜庢妸 SC1336 DVP 鐩告満鐨勮棰戦?佸叆 FPGA锛岀粡杩? DDR 缂撳瓨鍜屽熀纭? ISP 澶勭悊鍚庤緭鍑? HDMI銆俁K3588 閫氳繃 HDMI 閲囬泦璁惧鎺ユ敹 FPGA 杈撳嚭鐨勮棰戞祦锛屽苟浣跨敤 YOLO11-pose RKNN 妯″瀷妫?娴嬫按琛ㄥ皬琛ㄧ洏鐨勪腑蹇冪偣涓庢寚閽堝皷绔紝杩涗竴姝ヨ绠楁寚閽堣搴︺??

褰撳墠闃舵閲嶇偣瑙ｅ喅涓や釜闂锛?

1. FPGA 杈撳嚭鍥惧儚鍋忔殫銆佺櫧骞宠　涓嶇ǔ瀹氾紝闇?瑕侀?氳繃 UART 鍦ㄧ幇鍦哄姩鎬佽皟鑺? RGB 澧炵泭銆佸叏灞?澧炵泭鍜屼寒搴﹀亸缃??
2. RK3588 鎸囬拡瑙掑害妫?娴嬪瓨鍦ㄥ井灏忔尝鍔紝闇?瑕侀伩鍏嶅皬璇樊鎸佺画鍒锋柊鏄剧ず锛屽噺灏戣搴︽枃瀛楀拰鎸囬拡绾挎姈鍔ㄣ??

## 2. 鎬讳綋鏋舵瀯

```text
SC1336 DVP Camera
    |
    | 8-bit RAW / PCLK / VSYNC / HREF
    v
FPGA XYCrop
    |
    | RAW8
    v
DDR3 Frame Buffer
    |
    | RAW8 read at HDMI pixel clock
    v
RAW8 -> RGB888 Demosaic
    |
    v
UART Adjustable AWB / Brightness
    |
    v
Optional Frame Boundary Crop
    |
    v
HDMI 1280x720@60
    |
    v
RK3588 HDMI Capture /dev/video73
    |
    v
YOLO11-pose RKNN
    |
    v
Center / Tip / Angle / Stable Display
```

褰撳墠 FPGA 璁捐浠嶇劧淇濇寔 DDR 涓瓨鏀? RAW8 鏁版嵁锛孖SP 鏀惧湪 DDR 璇诲嚭涔嬪悗銆丠DMI 杈撳嚭涔嬪墠銆傝繖鏍峰彲浠ユ渶澶ч檺搴﹀鐢ㄥ師鏉ュ凡缁忕ǔ瀹氱殑鐏板害閾捐矾锛屽悓鏃朵究浜庨?愮骇鎵撳紑 ISP 鍔熻兘銆?

## 3. FPGA 瑙嗛閾捐矾

### 3.1 鏃堕挓鍩?

| 鏃堕挓 | 棰戠巼 | 涓昏鐢ㄩ?? |
| --- | --- | --- |
| `clk_ctrl` | 100 MHz | I2C銆乁ART銆佹帶鍒跺瘎瀛樺櫒 |
| `clk_cmos` | 24 MHz | SC1336 XCLK |
| `cmos_pclk` | 鐢辩浉鏈鸿緭鍑? | DVP 鏁版嵁閲囬泦 |
| `clk_video` / `lcd_clk` | 74.25 MHz | 1280x720 HDMI/LCD 杈撳嚭 |
| `clk_ddr_ref` | 250 MHz | MIG DDR3 绯荤粺鏃堕挓 |
| `clk_ddr_ref_200m` | 200 MHz | IDELAYCTRL 鍙傝?冩椂閽? |

UART 鍛戒护鍦? `clk_ctrl` 鍩熸帴鏀讹紝ISP 鍙傛暟鍦? `lcd_clk` 鍩熶娇鐢ㄣ?傝法鏃堕挓鍩熼噰鐢ㄤ袱绾у瘎瀛樺櫒鍚屾锛岄伩鍏嶆帶鍒朵俊鍙风洿鎺ヨ法鍩熼?犳垚浜氱ǔ鎬侀闄┿??

### 3.2 鏄剧ず妯″紡

椤跺眰鏂囦欢锛歚rtl/sc1336_hdmi.sv`

鏄剧ず妯″紡鐢? UART 鍛戒护鍒囨崲锛屾牳蹇冪姸鎬佷负锛?

```verilog
localparam [1:0] ISP_MODE_RAW  = 2'd0;
localparam [1:0] ISP_MODE_RGB  = 2'd1;
localparam [1:0] ISP_MODE_AWB  = 2'd2;
localparam [1:0] ISP_MODE_CROP = 2'd3;
```

杈撳嚭閫夋嫨閫昏緫锛?

```text
L mode       -> LCD test pattern, bypass DDR
D/X/G mode   -> DDR test pattern
S/R mode     -> sensor RAW gray
I mode       -> demosaic RGB
W mode       -> RGB + adjustable AWB / brightness
C mode       -> RGB + adjustable AWB / brightness + crop
```

杩欑妯″紡璁捐鐨勭洰鐨勪笉鏄竴娆℃?ц拷姹傛渶缁堢敾璐紝鑰屾槸鎶婇摼璺媶鎴愬彲楠岃瘉鐨勯樁娈碉細

1. 鍏堢‘璁? HDMI/LCD 鑷韩姝ｅ父銆?
2. 鍐嶇‘璁? DDR 鍐欒閾捐矾姝ｅ父銆?
3. 鍐嶇‘璁ょ浉鏈? RAW 鏁版嵁姝ｅ父銆?
4. 鍐嶇‘璁? Bayer 鐩镐綅鍜? RGB 杞崲姝ｅ父銆?
5. 鏈?鍚庤皟鐧藉钩琛°?佷寒搴﹀拰鍚庣画妫?娴嬫晥鏋溿??

## 4. UART 鍙皟 ISP 鍙傛暟

### 4.1 鍙傛暟瀹氫箟

褰撳墠鏂板鐨? ISP 鍙傛暟濡備笅锛?

| 鍙傛暟 | 榛樿鍊? | 瀹為檯鍚箟 |
| --- | --- | --- |
| `isp_r_gain` | 409 | 绾㈣壊閫氶亾绾? 1.60x |
| `isp_g_gain` | 256 | 缁胯壊閫氶亾 1.00x |
| `isp_b_gain` | 384 | 钃濊壊閫氶亾绾? 1.50x |
| `isp_all_gain` | 320 | 鍏ㄥ眬澧炵泭绾? 1.25x |
| `isp_brightness` | 8 | RGB 姣忛?氶亾鍔? 8 |

RGB 鍜屽叏灞?澧炵泭浣跨敤 Q8.8 鏍煎紡锛?

```text
256 = 1.00x
320 = 1.25x
384 = 1.50x
409 = 1.60x
512 = 2.00x
768 = 3.00x
```

璋冭妭鑼冨洿锛?

| 椤圭洰 | 鏈?灏忓?? | 鏈?澶у?? | 姝ヨ繘 |
| --- | --- | --- | --- |
| RGB / 鍏ㄥ眬澧炵泭 | 128锛岀害 0.50x | 768锛岀害 3.00x | 16锛岀害 0.0625x |
| 浜害鍋忕疆 | 0 | 96 | 4 |

### 4.2 璁＄畻娴佺▼

姣忎釜鍍忕礌鐨? RGB 澶勭悊娴佺▼涓猴細

```text
R1 = R0 * R_GAIN / 256
G1 = G0 * G_GAIN / 256
B1 = B0 * B_GAIN / 256

R2 = R1 * ALL_GAIN / 256
G2 = G1 * ALL_GAIN / 256
B2 = B1 * ALL_GAIN / 256

R_OUT = saturate(R2 + BRIGHTNESS)
G_OUT = saturate(G2 + BRIGHTNESS)
B_OUT = saturate(B2 + BRIGHTNESS)
```

鍏朵腑 `saturate()` 琛ㄧず楗卞拰鍒? 8 bit 鑼冨洿锛?

```text
value < 255 -> value
value >= 255 -> 255
```

### 4.3 娴佹按绾胯璁?

鐧藉钩琛¤绠楀凡缁忓仛鎴? `lcd_clk` 鍩熸祦姘寸嚎锛岄伩鍏嶅湪 74.25 MHz 涓嬪舰鎴愯繃闀跨粍鍚堣矾寰勩??

褰撳墠娴佹按绾块樁娈碉細

1. RGB 鍒嗛?氶亾涔樹互鍚勮嚜閫氶亾澧炵泭銆?
2. 涓婁竴绾х粨鏋滀箻浠ュ叏灞?澧炵泭銆?
3. 鍙栫缉鏀惧悗鐨? 12 bit 涓棿鍊笺??
4. 鍔犱寒搴﹀亸缃苟楗卞拰鍒? 8 bit銆?

杩欎細甯︽潵鍑犱釜鍍忕礌鏃堕挓鐨勫浐瀹氬欢杩燂紝浣嗗杩炵画瑙嗛鏄剧ず鍜屽悗缁? RK 妫?娴嬪奖鍝嶅緢灏忋?傜浉瀵规敹鐩婃槸闄嶄綆瀹炵幇鏃跺簭鍘嬪姏锛屽挨鍏舵槸鍦ㄥ伐绋嬪凡鏈? DDR銆両LA銆丠DMI 绛夊鏉傞?昏緫鐨勬儏鍐典笅鏇寸ǔ銆?

## 5. 鏇濆厜闄愬埗涓庢殫鍏夐棶棰?

SC1336 鍒濆鍖栨枃浠讹細`rtl/i2c_sc1336_config.v`

褰撳墠宸茬粡闄愬埗鏇濆厜鏃堕棿锛岄伩鍏嶆殫鍏変笅鐩告満鑷姩鎷夐暱鏇濆厜瀵艰嚧涓ラ噸鎷栧奖鍜屽欢鏃躲?傞檺鍒舵洕鍏夌殑浠ｄ环鏄殫鍦虹敾闈細鏇存殫锛屽洜姝ゅ悗缁寒搴﹁ˉ鍋垮簲浼樺厛閫氳繃锛?

1. FPGA 鍏ㄥ眬澧炵泭 `a/z`
2. FPGA 浜害鍋忕疆 `+/-`
3. 鍚堢悊鐨勫閮ㄨˉ鍏?
4. 鍚庣画 Gamma / 瀵规瘮搴︽槧灏?

涓嶅缓璁负浜嗘殫鍏変寒搴︾洿鎺ユ斁寮?鏇濆厜涓婇檺锛屽洜涓烘按琛ㄦ娴嬮渶瑕佸疄鏃舵?у拰绋冲畾杈圭紭锛岄暱鏇濆厜浼氶?犳垚鎸囬拡杩愬姩鏃舵嫋褰憋紝鏈?缁堝奖鍝嶈搴﹁绠椼??

## 6. RK3588 鎸囬拡妫?娴嬮摼璺?

### 6.1 杈撳叆瑙嗛

RK3588 閫氳繃 V4L2/GStreamer 璇诲彇 HDMI 閲囬泦璁惧锛屽綋鍓嶉粯璁よ澶囷細

```text
/dev/video73
```

榛樿杈撳叆瑙勬牸锛?

```text
1280x720 @ 60 fps
BGR raw frame
```

### 6.2 妯″瀷

褰撳墠 Web 鍚姩鑴氭湰榛樿浣跨敤 YOLO11-pose RKNN FP 妯″瀷锛?

```text
/home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn
```

璇ユā鍨嬪叧閿偣瀹氫綅鏇寸ǔ瀹氾紝閫傚悎鎸囬拡瑙掑害銆佸湀鏁板拰 m鲁 璇绘暟鑱旇皟銆傚疄娴? Web 閾捐矾绾? 16 鍒? 18 FPS锛屽崟甯ф帹鐞嗙害 48 ms銆?

蹇?熸ā寮忎娇鐢? YOLO11-pose RKNN INT8 hybrid 妯″瀷锛?

```text
/home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_headrs_float_normal.rknn
```

鍚姩鏂瑰紡锛?

```bash
WM_MODEL_MODE=fast ./run_hdmi_yolo11_pose_web.sh
```

INT8 妯″紡甯х巼鏇撮珮锛學eb 閾捐矾绾? 27 鍒? 28 FPS锛屽崟甯ф帹鐞嗙害 25 鍒? 28 ms锛屼絾鐢ㄦ埛瀹炴祴闈欐?佸叧閿偣绋冲畾鎬т笉濡? FP銆傜幇鍦烘寮忚瀵熸寚閽堣搴︽椂锛屼紭鍏堜娇鐢? FP锛涢渶瑕侀珮甯х巼棰勮鏃跺啀鍒囨崲 INT8銆?

鏈鍙敤鐨? INT8 妯″瀷涓嶆槸鏅?氬叏 INT8 閲忓寲锛岃?屾槸鎵嬪姩 hybrid 閲忓寲銆傛櫘閫? INT8銆乴etterbox INT8銆丮MSE INT8銆乤uto-hybrid normal 閮芥浘鍑虹幇鍒嗙被缃俊搴﹀拰鍏抽敭鐐圭疆淇″害鍏ㄤ负 0 鐨勯棶棰樸?傛渶缁堝彲鐢ㄧ増鏈繚鎶や簡 YOLO11-pose 瑙ｇ爜杈撳嚭澶达細

```text
/model.23/Concat_4_output_0_rs -> output0
```

杩欐牱 backbone/neck 澶ч儴鍒嗕粛鍙? INT8 鍔犻?燂紝杈撳嚭澶翠繚鎸佽冻澶熺簿搴︼紝閬垮厤妫?娴嬬粨鏋滆閲忓寲鍘嬪潖銆?

妫?娴嬭緭鍑哄寘鍚細

| 杈撳嚭 | 鍚箟 |
| --- | --- |
| `box` | 灏忚〃鐩樻娴嬫 |
| `score` | 妫?娴嬬疆淇″害 |
| `cls_id` | 灏忚〃鐩樼被鍒? |
| `kpts[0]` | 琛ㄧ洏涓績鐐? |
| `kpts[1]` | 鎸囬拡灏栫 |

绫诲埆鏍囩锛?

```python
LABELS = ["10^-1", "10^-2", "10^-3", "10^-4"]
```

### 6.3 瑙掑害璁＄畻

瑙掑害鐢变腑蹇冪偣鍒版寚閽堝皷绔殑鍚戦噺璁＄畻锛?

```python
angle = atan2(tip_y - center_y, tip_x - center_x)
```

杈撳嚭瑙掑害鑼冨洿杞崲涓猴細

```text
0 <= angle < 360
```

褰撳墠瑙掑害瀹氫箟涓哄浘鍍忓潗鏍囩郴瑙掑害锛?

| 鏂瑰悜 | 瑙掑害 |
| --- | --- |
| 鍚戝彸 | 0 搴? |
| 鍚戜笅 | 90 搴? |
| 鍚戝乏 | 180 搴? |
| 鍚戜笂 | 270 搴? |

姘磋〃鍥涗釜灏忚〃鐩樻槸鍗佽繘鍒惰繘浣嶅叧绯伙紝涓嶈兘鎶婂洓涓〃鐩樼殑鐬椂瑙掑害鎴栫疮璁″湀鏁扮洿鎺ョ浉鍔犮?傚綋鍓? Web 瀹炵幇閲囩敤鈥滄祴閲忚捣鐐? + 涓昏〃鐩樿繛缁浆瑙掆?濈殑鏂瑰紡璁＄畻鏈缁忚繃姘撮噺锛?

```text
鐐瑰嚮鈥滃紑濮嬫祴閲?/娓呴浂鈥?
    -> 娓呴浂鍚勮〃鐩樼疮璁″湀鏁?
    -> 鍚庣画鎸夎繛缁搴﹀樊绱 turns
    -> 鏈〃浼扮畻姘撮噺 = turns x volume_per_turn
```

姣忎釜琛ㄧ洏鐨勪綋绉惈涔夛細

| 琛ㄧ洏 | 姣忓皬鏍? | 姣忓湀姘撮噺 |
| --- | --- | --- |
| `10^-1` | 0.1 m鲁 | 1.0 m鲁 |
| `10^-2` | 0.01 m鲁 | 0.1 m鲁 |
| `10^-3` | 0.001 m鲁 | 0.01 m鲁 |
| `10^-4` | 0.0001 m鲁 | 0.001 m鲁 |

褰撳墠涓绘樉绀洪粯璁や娇鐢? `10^-4` 琛ㄧ洏浣滀负娴嬮噺鏉ユ簮锛屽洜涓哄畠鏈?鐏垫晱锛岄?傚悎鐭椂闂磋川妫?銆傚叾浠栬〃鐩樼殑鐙珛浼扮畻鍊间粛淇濈暀鍦ㄥ彸渚р?滆〃鐩樻娴嬪垪琛ㄢ?濓紝鍚庣画鍙敤浜庝竴鑷存?ф鏌ャ?佸紓甯歌烦鍙樺垽鏂拰杩涗綅鏍￠獙銆?

濡傛灉鍚庣画瑕佸仛姝ｅ紡姘磋〃鍚堟牸鍒ゅ畾锛岃繕闇?瑕佹牴鎹瘡涓皬琛ㄧ洏鐨勫疄闄呴浂鐐规柟鍚戙?侀『閫嗘椂閽堝叧绯汇?佽繘浣嶅叧绯汇?佸厑璁歌宸拰浜у搧鏍囧噯鍐嶅仛鏍囧畾鏄犲皠銆?

## 7. 瑙掑害鏄剧ず绋冲畾鍣?

淇敼鏂囦欢锛?

- `/home/demo/water_meter/code/hdmi_yolo11_pose_detect.py`
- `/home/demo/water_meter/code/hdmi_yolo11_pose_web.py`

鏂板绫伙細

```python
AngleStabilizer
```

璁捐鐩爣锛氭ā鍨嬪叧閿偣鍦ㄩ潤姝㈢敾闈笅鍙兘鏈? 1 鍒? 2 鍍忕礌璺冲姩锛岀洿鎺ユ樉绀轰細瀵艰嚧瑙掑害鏂囧瓧鍜屾寚閽堢嚎鎸佺画鎶栧姩銆傚洜姝ゆ樉绀哄眰澧炲姞姝诲尯鍜屼綆閫氭洿鏂般??

### 7.1 鐜舰瑙掑害宸?

瑙掑害鏄幆褰㈠彉閲忥紝涓嶈兘鐩存帴鐢ㄦ櫘閫氬噺娉曘?備緥濡? 359 搴﹀埌 1 搴︾殑鐪熷疄宸?兼槸 2 搴︼紝涓嶆槸 358 搴︺??

绋冲畾鍣ㄤ娇鐢細

```python
diff = (new_angle - old_angle + 180.0) % 360.0 - 180.0
```

寰楀埌鑼冨洿涓? `[-180, 180)` 鐨勬渶鐭搴﹀樊銆?

### 7.2 姝诲尯

濡傛灉瑙掑害鍙樺寲灏忎簬闃堝?硷細

```text
abs(diff) < angle_deadband
```

鍒欎繚鎸佷笂涓?甯ф樉绀鸿搴︿笉鍙樸?傞粯璁わ細

```text
angle_deadband = 1.0 deg
```

### 7.3 骞虫粦

濡傛灉瑙掑害鍙樺寲瓒呰繃姝诲尯锛屽垯鎸夋瘮渚嬫洿鏂帮細

```text
stable_angle = old_angle + alpha * diff
```

榛樿锛?

```text
angle_alpha = 0.30
```

`alpha` 瓒婂ぇ锛屾樉绀哄搷搴旇秺蹇紱`alpha` 瓒婂皬锛屾樉绀鸿秺绋充絾婊炲悗鏇存槑鏄俱??

## 8. 鐜版湁闄愬埗

1. FPGA 渚у綋鍓嶆槸鎵嬪姩鐧藉钩琛★紝涓嶆槸鑷姩鐧藉钩琛°??
2. FPGA 渚т寒搴︽彁鍗囧彲鑳介?犳垚楂樺厜楗卞拰锛岄渶瑕佺粨鍚堟按琛ㄨ〃鐩樿竟缂樻竻鏅板害璋冨弬銆?
3. RGB 鏁版嵁鏄湪 DDR 璇诲嚭鍚庣敓鎴愶紝DDR 鍐呬粛鐒朵繚瀛? RAW8銆?
4. RK 渚ц搴︾ǔ瀹氬櫒鍙ǔ瀹氭樉绀虹粨鏋滐紝涓嶆敼鍙樻ā鍨嬬湡瀹炶緭鍑恒??
5. FP 妯″紡鍏抽敭鐐规洿鍑嗕絾甯х巼浣庯紝INT8 hybrid 妯″紡甯х巼楂樹絾鍏抽敭鐐瑰彲鑳芥湁鍋忕Щ锛岄渶瑕佹牴鎹祴璇曠洰鏍囬?夋嫨銆?
6. 褰撳墠 m鲁 璇绘暟閲囩敤榛樿涓昏〃鐩? `10^-4` 鐨勮繛缁浆瑙掍及绠楋紝鐢ㄤ簬鏄剧ず鍜岃仈璋冿紱姝ｅ紡璐ㄦ浠嶉渶瑕佺粨鍚堜骇鍝佹爣鍑嗗仛闆剁偣銆佹柟鍚戙?佽繘浣嶅拰璇樊瀹瑰繊鏍囧畾銆?

## 9. 鍚庣画寤鸿

1. FPGA 鍔犲叆 Gamma 鎴栧垎娈典寒搴︽槧灏勶紝鐢ㄤ簬鏆楅儴澧炲己锛屽悓鏃堕檺鍒堕珮鍏夋孩鍑恒??
2. 澧炲姞鑷姩鏇濆厜/鑷姩澧炵泭绛栫暐锛屼絾蹇呴』闄愬埗鏈?澶ф洕鍏夋椂闂达紝閬垮厤鏆楀厜寤舵椂銆?
3. 鍦? RK 渚у缓绔嬭〃鐩樿鏁版ā鍨嬶細瑙掑害闆剁偣鏍囧畾銆佹柟鍚戞爣瀹氥?佸湀鏁拌繘浣嶉?昏緫銆佸紓甯歌烦鍙樿繃婊ゃ??
4. 閲囬泦澶氱鍏夌収銆佽搴﹀拰姘磋〃鍨嬪彿鏁版嵁锛屾彁鍗? YOLO11-pose 娉涘寲鑳藉姏銆?
5. C++ 鍏ㄩ摼璺缓璁粠鐜版湁 `cpp_bench` 鎵╁睍锛岄?愭鍔犲叆 V4L2/HDMI 閲囬泦銆佸墠澶勭悊銆乊OLO-pose 鍚庡鐞嗐?乄eb JSON/MJPEG 杈撳嚭銆?

## 10. 楠岃瘉璁板綍

宸插畬鎴愮殑楠岃瘉锛?

| 椤圭洰 | 缁撴灉 |
| --- | --- |
| Vivado RTL elaboration | 閫氳繃锛?0 Errors |
| RK Python 璇硶妫?鏌? | `py_compile` 閫氳繃 |
| RK 宸茬煡楠岃瘉鍥炬娴? | 妫?鍑? 4 涓皬琛ㄧ洏 |
| 瑙掑害绋冲畾鍙傛暟 | 鏀寔姝诲尯銆佽繛缁抚纭銆佸湀鏁扮疮璁℃鍖? |
| INT8 hybrid RKNN | 宸茬煡楠岃瘉鍥炬鍑? 4 涓皬琛ㄧ洏 |
| RK Web 瀹炴椂妫?娴? FP | HDMI 瀹炴椂娴佺害 16 鍒? 18 FPS锛屽叧閿偣鏇寸ǔ |
| RK Web 瀹炴椂妫?娴? INT8 | HDMI 瀹炴椂娴佺害 27 鍒? 28 FPS锛岄?傚悎楂樺抚鐜囬瑙? |
| C++ RKNN benchmark | 鐙崰 NPU 绾? 36 鍒? 38 FPS |

Vivado 妫?鏌ヤ腑浠嶅瓨鍦ㄤ竴浜涘凡鏈? IP/XDC warning 鍜? critical warning锛屾湭鍙戠幇鐢辨湰娆? ISP 涓插彛璋冨弬閫昏緫寮曞叆鐨勮娉曢敊璇??

## 11. RK3588 Web 閮ㄧ讲涓庤繍琛?

Web 涓荤▼搴忥細

```text
/home/demo/water_meter/code/hdmi_yolo11_pose_web.py
```

鍚姩鑴氭湰锛?

```text
/home/demo/water_meter/run_hdmi_yolo11_pose_web.sh
```

鍚姩鍛戒护锛?

```bash
cd /home/demo/water_meter
./run_hdmi_yolo11_pose_web.sh
```

娴忚鍣ㄨ闂細

```text
http://<RK3588-IP>:6008/
```

榛樿鍚姩鍙傛暟锛?

```bash
--device /dev/video73
--width 1280
--height 720
--fps 60
--conf 0.25
--core-mask all
--stream-width 1280
--stream-fps 12
--jpeg-quality 90
--angle-deadband 3.0
--angle-alpha 0.20
--angle-confirm-frames 4
--angle-confirm-band 2.0
--turn-deadband 4.0
```

Web 椤甸潰鏄剧ず鍐呭锛?

| 椤圭洰 | 鍚箟 |
| --- | --- |
| 瀹炴椂鐢婚潰 | HDMI 閲囬泦甯т笌妫?娴嬭鐩栧眰 |
| 瑙嗛鍐呯煭鏍囩 | 浠呮樉绀? `10^-1`銆乣10^-2` 绛夌煭鏍囩锛岄伩鍏嶉伄鎸＄洰鏍? |
| 鎺ㄧ悊 FPS | RKNN 妯″瀷鎺ㄧ悊甯х巼 |
| 鏄剧ず FPS | Web MJPEG 杈撳嚭甯х巼 |
| 妫?娴嬭?楁椂 | 鍗曞抚 RKNN 鎺ㄧ悊鑰楁椂 |
| 琛ㄧ洏瑙掑害 | 鍙充晶鏄剧ず鍥涗釜灏忚〃鐩樼殑鍘熷瑙掑害鍜岀ǔ瀹氳搴? |
| 鍦堟暟 | 鏍规嵁瑙掑害杩炵画鍙樺寲浼扮畻鐨勭疮璁″湀鏁? |
| 鏈缁忚繃姘撮噺 | 榛樿鐢? `10^-4` 琛ㄧ洏绱鍦堟暟 x 姣忓湀姘撮噺寰楀埌 |
| 鏈〃浼扮畻姘撮噺 | 姣忎釜琛ㄧ洏鐙珛缁欏嚭鐨勭粡杩囨按閲忎及绠楋紝涓嶅仛鍥涜〃鐩稿姞 |
| 浼扮畻鎬昏鏁? | 璧峰/鍩哄噯 m鲁 + 鏈缁忚繃姘撮噺 |
| 鍩哄噯 m鲁 | 鎵嬪姩杈撳叆鐨勫紑濮嬫祴閲忔椂鏈烘璇绘暟 |
| 寮?濮嬫祴閲?/娓呴浂 | 浠庡綋鍓嶈搴﹀紑濮嬬粺璁＄粡杩囨按閲? |
| 褰撳墠璁鹃浂浣? | 灏嗗綋鍓嶈搴﹁涓哄悇琛ㄧ洏闆剁偣锛屽苟閲嶆柊寮?濮嬫祴閲? |
| 瑙嗛妗嗗搴? | 鍙敼鍙樻祻瑙堝櫒鏄剧ず澶у皬锛屼笉鏀瑰彉鎺ㄦ祦鐮佺巼 |
| 鎺ㄦ祦瀹藉害 | 鍔ㄦ?佹敼鍙? MJPEG 杈撳嚭瀹藉害 |
| JPEG 璐ㄩ噺 | 鍔ㄦ?佹敼鍙? MJPEG 鍘嬬缉璐ㄩ噺 |
| 缃戦〉甯х巼 | 鍔ㄦ?佹敼鍙? MJPEG 杈撳嚭甯х巼 |

Web 鎺у埗鎺ュ彛锛?

```text
GET  /status       鑾峰彇鎺ㄧ悊銆佽鏁般?佹ā鍨嬨?佹帹娴佸弬鏁板拰閿欒鐘舵??
GET  /stream       MJPEG 瀹炴椂瑙嗛娴?
GET  /snapshot.jpg 褰撳墠杈撳嚭甯ф埅鍥?
POST /control      鏆傚仠銆佹爣娉ㄥ紑鍏炽?侀浂鐐规爣瀹氥?佹ā鍨嬪垏鎹€?佹帹娴佸弬鏁拌缃?
```

`/control` 鏀寔鐨勪富瑕佸姩浣滐細

| action | 浣滅敤 |
| --- | --- |
| `set_pause` | 鏆傚仠鎴栫户缁噰闆嗗拰鎺ㄧ悊 |
| `set_overlay` | 鏄剧ず鎴栭殣钘忔娴嬭鐩栧眰 |
| `set_stream` | 鍔ㄦ?佽缃? `stream_width`銆乣jpeg_quality`銆乣stream_fps` |
| `restart_model_mode` | 鍦? `accuracy` FP 鍜? `fast` INT8 闂村垏鎹? |
| `reset_turns` | 浠庡綋鍓嶈搴﹀紑濮嬫祴閲忥紝娓呴浂鏈缁忚繃姘撮噺 |
| `calibrate_zero` | 灏嗗綋鍓嶈搴﹁涓洪浂浣? |
| `set_base_m3` | 璁剧疆寮?濮嬫祴閲忔椂鐨? m鲁 鍩哄噯璇绘暟 |

甯哥敤缁存姢鍛戒护锛?

```bash
tail -f /home/demo/water_meter/hdmi_yolo11_pose_web.log
cat /home/demo/water_meter/hdmi_yolo11_pose_web.pid
fuser /dev/video73
fuser -k /dev/video73
```

## 12. INT8 妯″瀷楠岃瘉鏂规硶

宸茬煡楠岃瘉鍥撅細

```text
/home/demo/water_meter/pose_known_val.jpg
```

楠岃瘉鍛戒护锛?

```bash
cd /home/demo/water_meter/code
python3 -u ./hdmi_yolo11_pose_detect.py \
  --model /home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_headrs_float_normal.rknn \
  --image /home/demo/water_meter/pose_known_val.jpg \
  --save-output /home/demo/water_meter/int8_headrs_known.jpg \
  --conf 0.05 \
  --input-layout nhwc \
  --input-dtype uint8 \
  --color rgb
```

褰撳墠宸查獙璇佽緭鍑猴細

```text
detections=4
infer_ms=28.69
```

濡傛灉妫?娴嬫暟閲忎负 0锛屼紭鍏堟鏌ワ細

1. 妯″瀷鏄惁涓? `water_meter_yolo11n_pose_int8_headrs_float_normal.rknn`銆?
2. 杈撳叆鍙傛暟鏄惁涓? `--input-layout nhwc --input-dtype uint8 --color rgb`銆?
3. 鏄惁璇敤浜嗘櫘閫? INT8 妯″瀷銆?
4. 宸茬煡鍥剧墖鏄惁瀛樺湪銆?

## 13. C++ Benchmark 涓庡悗缁? C++ 鍖?

褰撳墠 C++ benchmark 鐩綍锛?

```text
/home/demo/water_meter/cpp_bench
```

鍙墽琛屾枃浠讹細

```text
/home/demo/water_meter/cpp_bench/rknn_pose_bench
```

鏈湴婧愮爜澶囦唤锛?

```text
rk3588/native/rknn_pose_bench.cpp
```

缂栬瘧鍛戒护锛?

```bash
cd /home/demo/water_meter/cpp_bench
g++ -O3 -std=c++17 rknn_pose_bench.cpp -I. -L/usr/lib -lrknnrt -o rknn_pose_bench
```

娴嬭瘯鍛戒护锛?

```bash
LD_LIBRARY_PATH=/usr/lib:/home/demo/water_meter/module \
./rknn_pose_bench \
  /home/demo/water_meter/module/int8_variants/water_meter_yolo11n_pose_int8_headrs_float_normal.rknn \
  300 20 1
```

鐙崰 NPU benchmark 缁撴灉锛?

| 妯″紡 | 骞冲潎鑰楁椂 | FPS |
| --- | --- | --- |
| `want_float=1` | 绾? 27.15 ms | 绾? 36.8 FPS |
| `want_float=0` | 绾? 26.51 ms | 绾? 37.7 FPS |

娉ㄦ剰锛氬鏋? Web 鏈嶅姟鍚屾椂杩愯锛孋++ benchmark 浼氬拰 Web 浜夌敤 NPU锛屽钩鍧囪?楁椂浼氳鎷夐珮銆傛祴璇? C++ 鏋侀檺鎬ц兘鍓嶅簲鍏堝仠姝? Web銆?

鍚庣画 C++ 鍏ㄩ摼璺缓璁垎涓夋瀹炵幇锛?

1. C++ V4L2/GStreamer 閲囬泦 `/dev/video73`锛屽畬鎴? 1280x720 BGR 鍒? 640x640 RGB/NHWC 鐨? letterbox 鍓嶅鐞嗐??
2. 澶嶇敤 RKNN C++ 鎺ㄧ悊锛屽姞鍏? YOLO11-pose 杈撳嚭瑙ｆ瀽銆丯MS銆佽搴︺?佸湀鏁板拰 m鲁 璁＄畻銆?
3. 浠? HTTP/WebSocket/鍏变韩鍐呭瓨鏂瑰紡鎶? JPEG 鍥惧儚鍜? JSON 鐘舵?佽緭鍑虹粰鐜版湁 Web 鍓嶇銆?
