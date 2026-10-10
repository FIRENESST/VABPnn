package cn.vabpnn.mod.mixin.client;
import com.mojang.blaze3d.platform.InputConstants;
import net.minecraft.client.KeyMapping;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(KeyMapping.class)
public interface KeyMappingAccess {
    @Accessor("key") InputConstants.Key vabpnn$boundKey();
    @Accessor("isDown") void vabpnn$forceDown(boolean value);
    @Accessor("clickCount") void vabpnn$setClickCount(int count);
}
