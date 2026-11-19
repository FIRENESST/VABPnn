package cn.vabpnn.mod.client;

import cn.vabpnn.mod.VabpnnMod;
import cn.vabpnn.mod.client.capture.CaptureController;
import net.fabricmc.api.ClientModInitializer;

public final class VabpnnClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        CaptureController.initialize();
        VabpnnMod.LOGGER.info("VABPnn {}: GPU PNN -> JEPA -> Transformer action decoder", VabpnnMod.VERSION);
        VabpnnMod.LOGGER.info("VABPnn client initialized.");
    }
}