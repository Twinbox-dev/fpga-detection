import serial
import time

# 配置串口（与你的 api.py 一致）
com = serial.Serial('/dev/XIPAN', 9600, timeout=1)
time.sleep(0.1)  # 等待串口稳定

# 抓取和释放指令（十六进制字符串转字节）
zhua_qu = bytes.fromhex('A0 01 01 A2')
shi_fang = bytes.fromhex('A0 01 00 A1')

print("启动吸盘...")
com.write(zhua_qu)
time.sleep(2)  # 保持吸盘开启 2 秒

print("释放吸盘...")
com.write(shi_fang)
time.sleep(0.5)

com.close()
print("测试完成。")