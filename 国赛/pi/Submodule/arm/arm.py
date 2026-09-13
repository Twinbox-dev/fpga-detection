import time
from Arm_Lib import Arm_Device

# 初始化机械臂
Arm = Arm_Device()
time.sleep(0.1)

# ---------- 定义点位（6个舵机，第6个固定90） ----------
p_initial = [90, 90, 90, 0, 90, 90]   # 复位位置

# 吸取位置（模拟吸盘下降）
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

speed = 800  # 运动时间（毫秒）

# 选择测试状态："OK" 或 "NG"
flags = "OK"   # 可改为 "OK" 测试

print(f"=== 模拟吸取动作 ===")
# 1. 下降到吸取位置
Arm.Arm_serial_servo_write6(*p_Z, speed)
time.sleep(speed / 1000 + 0.3)
# 2. 抬高（模拟吸起）
Arm.Arm_serial_servo_write6(*p_Z_1, speed)
time.sleep(speed / 1000 + 0.3)

print(f"=== 执行 {flags} 状态运动 ===")
if flags == "OK":
    Arm.Arm_serial_servo_write6(*p_OK_1, speed)
    time.sleep(speed / 1000 + 0.3)
    Arm.Arm_serial_servo_write6(*p_OK_2, speed)
    time.sleep(speed / 1000 + 0.3)
    Arm.Arm_serial_servo_write6(*p_OK, speed)
    time.sleep(speed / 1000 + 0.3)
elif flags == "NG":
    Arm.Arm_serial_servo_write6(*p_NG_1, speed)
    time.sleep(speed / 1000 + 0.3)
    Arm.Arm_serial_servo_write6(*p_NG, speed)
    time.sleep(speed / 1000 + 0.3)
    
    # NG 时蜂鸣器响 1000ms（delay=10 代表 10×100ms = 1000ms）
    print("NG → 蜂鸣器响 1 秒")
    Arm.Arm_Buzzer_On(delay=10)
    time.sleep(1.2)  # 稍长于蜂鸣时间，确保听到

# 等待 2 秒后复位
print("等待 2 秒后复位...")
time.sleep(2)
Arm.Arm_serial_servo_write6(*p_initial, speed)
time.sleep(speed / 1000 + 0.3)

print("运动完成，已复位。")
