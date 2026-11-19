package cn.vabpnn.telemetry;

import com.google.gson.*;
import net.fabricmc.fabric.api.client.gametest.v1.context.*;
import net.minecraft.core.*;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.inventory.*;
import net.minecraft.world.item.*;
import net.minecraft.world.entity.item.ItemEntity;
import net.minecraft.world.phys.*;
import net.minecraft.util.ProblemReporter;
import net.minecraft.world.level.storage.TagValueOutput;

final class ProgressionGameTest {
 static void check(boolean ok,String label){if(!ok)throw new AssertionError(label);}
 static void run(ClientGameTestContext context,TestServerContext server){
  server.runCommand("tp @p 0.5 -60 0.5");server.runCommand("setblock 1 -60 -1 minecraft:crafting_table");
  craft(context,server,Items.OAK_PLANKS,Items.WOODEN_AXE,false,1);
  select(context,server,Items.WOODEN_AXE);
  for(int z=0;z<3;z++)mine(context,server,new BlockPos(2,-60,z));
  server.runCommand("tp @p 2.5 -60 0.5");context.waitFor(mc->hasPickup("minecraft:oak_log",true),200);
  server.runCommand("tp @p 0.5 -60 0.5");
  server.runOnServer(s->{var p=s.getPlayerList().getPlayer(TelemetryMod.controlledPlayer);p.getMainHandItem().setDamageValue(p.getMainHandItem().getMaxDamage()-1);});
  context.waitFor(mc->mc.player.getMainHandItem().getDamageValue()>=mc.player.getMainHandItem().getMaxDamage()-1,100);
  mine(context,server,new BlockPos(2,-60,3));context.waitFor(mc->stat("broken_by_type","minecraft:wooden_axe")>=1,200);
  craft(context,server,Items.OAK_PLANKS,Items.WOODEN_AXE,false,2);select(context,server,Items.WOODEN_AXE);mine(context,server,new BlockPos(3,-60,0));
  craft(context,server,Items.COBBLESTONE,Items.STONE_AXE,false,1);
  craft(context,server,Items.IRON_INGOT,Items.IRON_PICKAXE,true,1);
  server.runOnServer(s->{var p=s.getPlayerList().getPlayer(TelemetryMod.controlledPlayer);for(int i=0;i<p.getInventory().getContainerSize();i++){ItemStack stack=p.getInventory().getItem(i);if(stack.is(Items.OAK_LOG)){ItemEntity dropped=p.drop(stack.split(1),false,true);check(dropped!=null,"redrop created an entity");dropped.setPos(p.getX(),p.getY(),p.getZ());dropped.setDeltaMovement(Vec3.ZERO);dropped.setNoPickUpDelay();return;}}throw new AssertionError("no harvested log available to redrop");});
  context.waitFor(mc->hasPickup("minecraft:oak_log",false),300);
  mergedPartialPickup(server);
  TelemetryMod.LOG.info("PASS: real crafting slots, first wooden axe/iron pickaxe, wear-out/recraft/use, stone axe upgrade, harvested pickup, redrop exclusion, merged and partial pickups, persisted provenance");
 }
 private static void craft(ClientGameTestContext context,TestServerContext server,Item material,Item result,boolean pickaxe,int expected){
  server.runCommand("tp @p 0.5 -60 0.5");
  context.waitFor(mc->Math.abs(mc.player.getX()-.5)<.1&&Math.abs(mc.player.getZ()-.5)<.1&&mc.level.getBlockState(new BlockPos(1,-60,-1)).is(net.minecraft.world.level.block.Blocks.CRAFTING_TABLE),200);
  context.waitTicks(3); // A same-position teleport still requires a client acknowledgement before interaction packets.
  context.runOnClient(mc->mc.gameMode.useItemOn(mc.player,InteractionHand.MAIN_HAND,new BlockHitResult(new Vec3(1.5,-59,-.5),Direction.UP,new BlockPos(1,-60,-1),false)));
  context.waitFor(mc->mc.player.containerMenu instanceof CraftingMenu,100);
  server.runOnServer(s->{var p=s.getPlayerList().getPlayer(TelemetryMod.controlledPlayer);var menu=p.containerMenu;for(int i=1;i<=9;i++)menu.getSlot(i).set(ItemStack.EMPTY);for(int i:pickaxe?new int[]{1,2,3}:new int[]{1,2,4})menu.getSlot(i).set(new ItemStack(material));for(int i:new int[]{5,8})menu.getSlot(i).set(new ItemStack(Items.STICK));menu.broadcastChanges();});
  context.waitFor(mc->mc.player.containerMenu.getSlot(0).getItem().is(result),100);
  context.runOnClient(mc->mc.gameMode.handleContainerInput(mc.player.containerMenu.containerId,0,0,ContainerInput.QUICK_MOVE,mc.player));
  String id=net.minecraft.core.registries.BuiltInRegistries.ITEM.getKey(result).toString();context.waitFor(mc->stat("crafted_by_type",id)>=expected,200);
  context.runOnClient(mc->mc.player.closeContainer());context.waitFor(mc->!(mc.player.containerMenu instanceof CraftingMenu),100);
 }
 private static void select(ClientGameTestContext context,TestServerContext server,Item item){
  server.runOnServer(s->{var p=s.getPlayerList().getPlayer(TelemetryMod.controlledPlayer);var inv=p.getInventory();for(int i=0;i<inv.getContainerSize();i++)if(inv.getItem(i).is(item)){ItemStack stack=inv.getItem(i),held=p.getMainHandItem();if(stack!=held){inv.setItem(i,held);p.setItemInHand(InteractionHand.MAIN_HAND,stack);}break;}});
  context.waitFor(mc->mc.player.getMainHandItem().is(item),100);
 }
 private static void mine(ClientGameTestContext context,TestServerContext server,BlockPos pos){
  server.runCommand("setblock "+pos.getX()+" "+pos.getY()+" "+pos.getZ()+" minecraft:oak_log");
  context.waitFor(mc->mc.level.getBlockState(pos).is(net.minecraft.world.level.block.Blocks.OAK_LOG),100);context.runOnClient(mc->mc.gameMode.startDestroyBlock(pos,Direction.UP));
  String position=pos.getX()+","+pos.getY()+","+pos.getZ();context.waitFor(mc->{mc.gameMode.continueDestroyBlock(pos,Direction.UP);return hasBreak(position);},300);context.runOnClient(mc->mc.gameMode.stopDestroyBlock());
 }
 private static boolean hasBreak(String pos){if(!TelemetryMod.lastPacket.has("events"))return false;for(JsonElement value:TelemetryMod.lastPacket.getAsJsonArray("events")){JsonObject e=value.getAsJsonObject();if(e.get("type").getAsString().equals("block_break")&&e.get("pos").getAsString().equals(pos)&&e.getAsJsonObject("tool").getAsJsonObject("harvest").getAsJsonObject("items").has("minecraft:oak_log"))return true;}return false;}
 private static boolean hasPickup(String item,boolean traced){if(!TelemetryMod.lastPacket.has("events"))return false;for(JsonElement value:TelemetryMod.lastPacket.getAsJsonArray("events")){JsonObject e=value.getAsJsonObject();if(e.get("type").getAsString().equals("item_pickup")&&e.get("item").getAsString().equals(item)&&(!e.getAsJsonObject("origins").isEmpty())==traced)return true;}return false;}
 private static int stat(String group,String item){JsonObject t=TelemetryMod.lastPacket;if(!t.has(group))return 0;JsonObject map=t.getAsJsonObject(group);return map.has(item)?map.get(item).getAsInt():0;}
 private static void mergedPartialPickup(TestServerContext server){
  server.runOnServer(s->{try{
   var p=s.getPlayerList().getPlayer(TelemetryMod.controlledPlayer);ItemEntity target=new ItemEntity(p.level(),p.getX(),p.getY(),p.getZ(),new ItemStack(Items.OAK_LOG,2)),source=new ItemEntity(p.level(),p.getX(),p.getY(),p.getZ(),new ItemStack(Items.OAK_LOG,3));
   ((TrackedDrop)target).scorer$origins().add("merge-a",2);((TrackedDrop)source).scorer$origins().add("merge-b",3);
   var merge=ItemEntity.class.getDeclaredMethod("merge",ItemEntity.class,ItemStack.class,ItemEntity.class,ItemStack.class);merge.setAccessible(true);merge.invoke(null,target,target.getItem(),source,source.getItem());
   check(target.getItem().getCount()==5&&source.isRemoved(),"real entity merge count");String origins=((TrackedDrop)target).scorer$origins().save();check(origins.contains("merge-a")&&origins.contains("merge-b"),"merge preserves both origins");
   TagValueOutput saved=TagValueOutput.createWithContext(ProblemReporter.DISCARDING,p.registryAccess());target.saveWithoutId(saved);check(saved.buildResult().toString().contains("vabpnn_scorer_origins"),"provenance included in real entity NBT");
   for(int i=0;i<36;i++)p.getInventory().setItem(i,new ItemStack(Items.COBBLESTONE,64));p.getInventory().setItem(0,new ItemStack(Items.OAK_LOG,63));target.setNoPickUpDelay();target.playerTouch(p);
   check(target.getItem().getCount()==4&&p.getInventory().getItem(0).getCount()==64,"one item fits in backpack");JsonObject remaining=JsonParser.parseString(((TrackedDrop)target).scorer$origins().save()).getAsJsonObject();check(remaining.get("merge-a").getAsInt()+remaining.get("merge-b").getAsInt()==4,"partial pickup consumes precisely one provenance unit");
   p.getInventory().setItem(1,ItemStack.EMPTY);target.playerTouch(p);check(target.isRemoved()&&((TrackedDrop)target).scorer$origins().save().equals("{}"),"full pickup clears provenance even when vanilla resets entity stack count");
   for(int i=0;i<36;i++)p.getInventory().setItem(i,ItemStack.EMPTY);
  }catch(ReflectiveOperationException e){throw new RuntimeException(e);}});
 }
}
