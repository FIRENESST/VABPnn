package cn.vabpnn.mod.mixin.client;
import net.minecraft.client.ToggleKeyMapping;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(ToggleKeyMapping.class)
public interface ToggleKeyMappingAccess {
    @Accessor("releasedByScreenWhenDown") void vabpnn$setScreenRestore(boolean value);
}
