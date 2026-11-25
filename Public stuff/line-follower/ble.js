export class BLEDevice {
    constructor() {
        this.device = null;
        this.server = null;
        this.service = null;
        this.writeCharacteristic = null;
        this.notifyCharacteristic = null;
        this.callback = null;
        this.disconnectCallback = null;
    }

    async scan() {
        const SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb';
        this._name = null;
        this.device = await navigator.bluetooth.requestDevice({
            filters: [{ services: [SERVICE_UUID] }]
        });
        console.log('BLE picked:', JSON.stringify({
            name: this.device.name, id: this.device.id
        }));
    }

    get name() {
        // '' (never null) so the Python side gets a plain string
        return this.device ? (this.device.name || this._name || '') : '';
    }

    // callback is set via .callback property before calling connect().
    // GATT setup is flaky right after the picker closes, so retry the whole
    // connect -> service -> characteristics -> notifications sequence.
    async connect(serviceUUID, writeUUID, notifyUUID, attempts = 5) {
        if (!this.device) {
            try {
                this.device = await navigator.bluetooth.requestDevice({
                    filters: [{ services: [serviceUUID] }]
                });
            } catch (error) {
                console.error('BLE picker error:', error);
                return false;
            }
        }
        if (!this._listening) {
            this.device.addEventListener(
                'gattserverdisconnected',
                this.handleDisconnect.bind(this)
            );
            this._listening = true;
        }
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

        for (let i = 1; i <= attempts; i++) {
            try {
                // Let the radio settle (longer on each retry)
                if (i > 1) await sleep(500 * i);   // first try goes straight in
                if (this.device.gatt.connected) this.device.gatt.disconnect();
                this.server  = await this.device.gatt.connect();
                this.service = await this.server.getPrimaryService(serviceUUID);
                this.writeCharacteristic  = await this.service.getCharacteristic(writeUUID);
                this.notifyCharacteristic = await this.service.getCharacteristic(notifyUUID);
                await this.notifyCharacteristic.startNotifications();
                this.notifyCharacteristic.addEventListener(
                    'characteristicvaluechanged',
                    this.handleNotification.bind(this)
                );
                console.log(`BLE connected: ${this.device.name} (attempt ${i})`);
                return true;
            } catch (error) {
                console.warn(`BLE connect attempt ${i}/${attempts} failed:`, error.name, error.message);
            }
        }
        console.error('BLE connect gave up after', attempts, 'attempts');
        try { if (this.device.gatt.connected) this.device.gatt.disconnect(); } catch (e) {}
        return false;
    }

    handleNotification(event) {
        const data = new Uint8Array(event.target.value.buffer);
        if (this.callback) {
            try { this.callback(data); }
            catch (err) { console.error('Notification callback error:', err); }
        }
    }

    handleDisconnect(event) {
        console.log('BLE disconnected:', this.device?.name);
        if (this.disconnectCallback) {
            try { this.disconnectCallback(event); }
            catch (err) { console.error('Disconnect callback error:', err); }
        }
    }

    async send(data) {
        if (!this.writeCharacteristic) throw new Error('Not connected');
        await this.writeCharacteristic.writeValue(new Uint8Array(data));
    }

    disconnect() {
        if (this.device?.gatt?.connected) {
            this.device.gatt.disconnect();
        }
    }
}
