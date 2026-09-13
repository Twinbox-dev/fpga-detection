import time
import serial
from Arm_Lib import Arm_Device

# ===================== 用户可配置参数 =====================
# 串口设备路径（吸盘控制）
SUCTION_SERIAL_PORT = '/dev/XIPAN'
SUCTION_SERIAL_BAUD = 9600

# 吸盘等待时间（秒）：到达吸取位置后，等待吸力达到最大再抬起
SUCTION_WAIT_TIME = 1.0

# 机械臂运动速度（毫秒）
ARM_SPEED = 800

# 测试状态：'OK' 或 'NG'
TEST_FLAGS = "OK"

# 蜂鸣器持续时间（单位：100ms，例如10=1000ms）
BUZZER_DELAY = 10
# ==========================================================

# ---------- 串口初始化（吸盘控制） ----------
com = serial.Serial(SUCTION_SERIAL_PORT, SUCTION_SERIAL_BAUD, timeout=1)
time.sleep(0.1)

# 吸盘指令
zhua_qu = bytes.fromhex('A0 01 01 A2')
shi_fang = bytes.fromhex('A0 01 00 A1')

def zhua_qu_xipan():
    """启动吸盘"""
    try:
        com.write(zhua_qu)
        time.sleep(0.1)
        print("吸盘已启动")
    except Exception as e:
        print(f"吸盘启动失败: {e}")

def shi_fang_xipan():
    """释放吸盘"""
    try:
        com.write(shi_fang)
        time.sleep(0.1)
        print("吸盘已释放")
    except Exception as e:
        print(f"吸盘释放失败: {e}")

# ---------- 机械臂初始化 ----------
Arm = Arm_Device()
time.sleep(0.1)

# ---------- 定义点位（6个舵机，第6个固定90） ----------
p_initial = [90, 90, 90, 0, 90, 90]   # 复位位置

# 吸取位置
p_Z = [90, 38, 26, 22, 90, 90]
# 吸取后抬高
p_Z_1 = [90, 90, 0, 90, 90, 90]

# OK 分支
p_OK_1 = [90, 90, 90, 0, 90, 90]
p_OK_2 = [-10, 90, 90, 0, 90, 90]
p_OK   = [-10, 65, 0, 30, 90, 90]

# NG 分支
p_NG_1 = [90, 90, 90, 0, 90, 90]
p_NG   = [180, 55, 11, 24, 90, 90]   # 原196改为180，避免越界

print(f"=== 开始测试，状态: {TEST_FLAGS} ===")

# ---------- 1. 模拟吸取动作 ----------
print("步骤1: 下降到吸取位置")
Arm.Arm_serial_servo_write6(*p_Z, ARM_SPEED)
time.sleep(ARM_SPEED / 1000 + 0.3)

# 启动吸盘
zhua_qu_xipan()

# 等待吸力达到最大
print(f"等待 {SUCTION_WAIT_TIME} 秒让吸力达到最大...")
time.sleep(SUCTION_WAIT_TIME)

print("步骤2: 抬高（模拟吸起）")
Arm.Arm_serial_servo_write6(*p_Z_1, ARM_SPEED)
time.sleep(ARM_SPEED / 1000 + 0.3)

# ---------- 2. 执行 OK/NG 运动 ----------
print(f"步骤3: 执行 {TEST_FLAGS} 运动")
if TEST_FLAGS == "OK":
    Arm.Arm_serial_servo_write6(*p_OK_1, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
    Arm.Arm_serial_servo_write6(*p_OK_2, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
    Arm.Arm_serial_servo_write6(*p_OK, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
elif TEST_FLAGS == "NG":
    Arm.Arm_serial_servo_write6(*p_NG_1, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
    Arm.Arm_serial_servo_write6(*p_NG, ARM_SPEED)
    time.sleep(ARM_SPEED / 1000 + 0.3)
    
    # NG 时蜂鸣器响
    print("NG → 蜂鸣器响")
    Arm.Arm_Buzzer_On(delay=BUZZER_DELAY)
    time.sleep(BUZZER_DELAY * 0.1 + 0.2)  # 略长于蜂鸣时间

# ---------- 3. 在目标位置释放吸盘 ----------
print("步骤4: 释放吸盘")
shi_fang_xipan()

# ---------- 4. 等待2秒后复位 ----------
print("等待 2 秒后复位...")
time.sleep(2)
Arm.Arm_serial_servo_write6(*p_initial, ARM_SPEED)
time.sleep(ARM_SPEED / 1000 + 0.3)

print("测试完成，已复位。")

# 关闭串口
com.close()
