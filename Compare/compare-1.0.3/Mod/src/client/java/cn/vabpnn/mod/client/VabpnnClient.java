package cn.vabpnn.mod.client;

import cn.vabpnn.mod.VabpnnMod;
import cn.vabpnn.mod.client.capture.CaptureController;
import net.fabricmc.api.ClientModInitializer;

public final class VabpnnClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        CaptureController.initialize();
        VabpnnMod.LOGGER.info("VABPnn compare-1.0.3: 7680 fully connected block nodes, 2 pair-bias heads, 118092470 Transformer parameters; trainable matrix memory gates; v11 checkpoint; optional acknowledged Scorer policy-head training; read-only previous-model migration.");
    }
}