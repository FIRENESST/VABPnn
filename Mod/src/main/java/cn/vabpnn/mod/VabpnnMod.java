package cn.vabpnn.mod;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Shared entry point: runs in both the client and dedicated server.
 * Keep client-only Minecraft classes out of this source set.
 */
public final class VabpnnMod implements ModInitializer {
    public static final String MOD_ID = "vabpnn";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("VABPnn mod initialized.");

        // In a development world, enter /vabpnn to check that this mod loaded.
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(Commands.literal(MOD_ID)
                .executes(context -> {
                    context.getSource().sendSuccess(
                        () -> Component.literal("VABPnn is loaded. Your Fabric mod is ready!"),
                        false
                    );
                    return 1;
                }))
        );
    }
}