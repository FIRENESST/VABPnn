package cn.vabpnn.mod.mixin.client;

import net.minecraft.client.main.Main;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Restore Windows AWT before Minecraft starts; vanilla Main explicitly sets headless=true. */
@Mixin(Main.class)
public abstract class DesktopScopeMixin {
    @Inject(method = "<clinit>", at = @At("TAIL"))
    private static void vabpnn$enableDesktopScope(CallbackInfo ci) {
        if (System.getProperty("os.name", "").startsWith("Windows")) {
            System.setProperty("java.awt.headless", "false");
        }
    }
}
