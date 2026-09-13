# alarm_serial.py
import serial
import time

class SerialAlarm:
    """最简串口报警器模块，发送十六进制指令触发报警"""

    def __init__(self, port='/dev/BAOJING', baudrate=9600, timeout=0.5):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._connect()

    def _connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except Exception:
            self.ser = None

    def alarm_on(self):
        """触发报警（发送一次指令）"""
        cmd = bytes.fromhex('7E FF 06 03 00 00 01 EF')
        try:
            if self.ser is None or not self.ser.is_open:
                self._connect()
            self.ser.write(cmd)
        except Exception:
            # 尝试重连后再次发送
            try:
                self.ser.close()
            except:
                pass
            self._connect()
            if self.ser and self.ser.is_open:
                self.ser.write(cmd)

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

if __name__ == "__main__":
    alarm = SerialAlarm()
    alarm.alarm_on()