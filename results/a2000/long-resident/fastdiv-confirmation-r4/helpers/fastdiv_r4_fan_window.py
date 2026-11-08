import json
import os
from pathlib import Path
import signal
import subprocess as S
import time

BASE = Path(__file__).parent
FAN = Path('/proc/acpi/ibm/fan')
WATCH = Path('/sys/bus/platform/drivers/thinkpad_hwmon/fan_watchdog')
STATUS = BASE / 'fastdiv-r4-fan-window-status.json'

def main():
    if os.geteuid() != 0:
        raise SystemExit('Administrator authentication required')
    previous = BASE/'fastdiv-r3-fan-window-status.json'
    wait_deadline=time.monotonic()+660
    while previous.exists() and json.loads(previous.read_text()).get('status')=='active':
        if time.monotonic()>wait_deadline:raise SystemExit('Previous bounded fan window did not finish')
        time.sleep(3)
    if not any(line.split() == ['level:', 'auto'] for line in FAN.read_text().splitlines()):
        raise SystemExit('Another manual fan policy is active')
    original = WATCH.read_text().strip()
    def gpu_limit():
        return float(S.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True).strip())
    original_power = gpu_limit()
    if abs(original_power-35.0)>0.01:
        raise SystemExit('Unexpected GPU power limit %s' % original_power)
    active = S.run(['systemctl', 'is-active', '--quiet', 'llm-fan.service']).returncode == 0
    record = {'started_unix': time.time(), 'duration_seconds': 1200, 'watchdog_seconds': 30,
              'original_watchdog': original, 'original_power_limit_W': original_power, 'power_cap_W': 25, 'service_was_active': active, 'samples': []}
    def save():
        STATUS.write_text(json.dumps(record, indent=2)+'\n')
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if active:
            S.run(['systemctl', 'stop', 'llm-fan.service'], check=True, timeout=15)
        FAN.write_text('watchdog 30\n')
        FAN.write_text('level 7\n')
        S.run(['nvidia-smi','-i','0','-pl','25'], check=True, timeout=15, capture_output=True)
        record['power_limit_readback_W'] = gpu_limit()
        if abs(record['power_limit_readback_W']-25.0)>0.01: raise RuntimeError('Power cap not applied')
        record['status'] = 'active'
        save()
        print('Normal maximum fans active for at most 1200 seconds', flush=True)
        deadline = time.monotonic()+1200
        while time.monotonic() < deadline and not (BASE/'fastdiv-r4-fan-window.stop').exists():
            FAN.write_text('watchdog 30\n')
            FAN.write_text('level 7\n')
            speeds = [int(p.read_text()) for h in Path('/sys/class/hwmon').glob('hwmon*')
                      if (h/'name').read_text().strip() == 'thinkpad' for p in sorted(h.glob('fan*_input'))]
            record['samples'].append({'time':time.time(), 'fan_rpm':speeds})
            save()
            time.sleep(3)
    finally:
        errors = []
        try:
            S.run(['nvidia-smi','-i','0','-pl',str(original_power)], check=True, timeout=15, capture_output=True)
            record['power_limit_restored_W'] = gpu_limit()
            if abs(record['power_limit_restored_W']-original_power)>0.01: errors.append('Power limit readback mismatch')
        except Exception as e: errors.append('Power limit restore failed: '+str(e))
        for command in ('level auto\n', 'watchdog '+original+'\n'):
            try: FAN.write_text(command)
            except OSError as e: errors.append(str(e))
        if active:
            restored = S.run(['systemctl', 'start', 'llm-fan.service'], timeout=15)
            if restored.returncode: errors.append('Fan service restart failed')
        record.update(status='restored' if not errors else 'restoration_failed', errors=errors,
                      finished_unix=time.time(), fan_readback=FAN.read_text())
        save()
        print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)

if __name__ == '__main__':
    main()
