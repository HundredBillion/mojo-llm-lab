import json
import os
from pathlib import Path
import signal
import subprocess as S
import time

BASE = Path(__file__).parent
FAN = Path('/proc/acpi/ibm/fan')
WATCH = Path('/sys/bus/platform/drivers/thinkpad_hwmon/fan_watchdog')
STATUS = BASE / 'coverage4-fan-window-status.json'

def main():
    if os.geteuid() != 0:
        raise SystemExit('Administrator authentication required')
    previous = BASE/'coverage3-fan-window-status.json'
    wait_deadline=time.monotonic()+660
    while previous.exists() and json.loads(previous.read_text()).get('status')=='active':
        if time.monotonic()>wait_deadline:raise SystemExit('Previous bounded fan window did not finish')
        time.sleep(3)
    if not any(line.split() == ['level:', 'auto'] for line in FAN.read_text().splitlines()):
        raise SystemExit('Another manual fan policy is active')
    original = WATCH.read_text().strip()
    def smi(*a):
        return S.run(['nvidia-smi',*a],capture_output=True,text=True,timeout=15)
    original_pm = smi('--query-gpu=persistence_mode','--format=csv,noheader').stdout.strip()
    if original_pm not in ('Enabled','Disabled'):
        raise SystemExit('Unexpected persistence mode %r' % original_pm)
    active = S.run(['systemctl', 'is-active', '--quiet', 'llm-fan.service']).returncode == 0
    record = {'started_unix': time.time(), 'duration_seconds': 1200, 'watchdog_seconds': 30,
              'original_watchdog': original, 'original_persistence_mode': original_pm, 'gpu_clock_lock_MHz': [210, 900], 'service_was_active': active, 'samples': []}
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
        pm = smi('-i','0','-pm','1'); lock = smi('-i','0','-lgc','210,900')
        record['persistence_set'] = {'rc': pm.returncode, 'out': (pm.stdout+pm.stderr).strip()[-300:]}
        record['clock_lock_set'] = {'rc': lock.returncode, 'out': (lock.stdout+lock.stderr).strip()[-300:]}
        if (pm.returncode or lock.returncode or 'GPU clocks set to' not in lock.stdout
                or smi('--query-gpu=persistence_mode','--format=csv,noheader').stdout.strip() != 'Enabled'):
            raise RuntimeError('Clock lock or persistence mode not applied')
        record['clock_lock_active'] = True
        record['status'] = 'active'
        save()
        print('Normal maximum fans active for at most 1200 seconds', flush=True)
        deadline = time.monotonic()+1200
        while time.monotonic() < deadline and not (BASE/'coverage4-fan-window.stop').exists():
            FAN.write_text('watchdog 30\n')
            FAN.write_text('level 7\n')
            speeds = [int(p.read_text()) for h in Path('/sys/class/hwmon').glob('hwmon*')
                      if (h/'name').read_text().strip() == 'thinkpad' for p in sorted(h.glob('fan*_input'))]
            record['samples'].append({'time':time.time(), 'fan_rpm':speeds})
            save()
            time.sleep(3)
    finally:
        errors = []
        for args in (('-i','0','-rgc'), ('-i','0','-pm','1' if original_pm=='Enabled' else '0')):
            try:
                r = smi(*args)
                if r.returncode: errors.append('GPU restore failed: '+' '.join(args)+': '+(r.stdout+r.stderr).strip()[-200:])
            except Exception as e: errors.append('GPU restore error: '+str(e))
        record['persistence_mode_restored'] = smi('--query-gpu=persistence_mode','--format=csv,noheader').stdout.strip()
        if record['persistence_mode_restored'] != original_pm: errors.append('Persistence mode readback mismatch')
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
