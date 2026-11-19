package cn.vabpnn.scorer;

import com.google.gson.*;

/** The collector reports facts; adjustable policy and rewards remain in Scorer. */
final class ToolScoring {
 record Result(String kind,String reason,double penalty,boolean appropriate){}
 static Result evaluate(JsonObject tool,JsonObject rules){
  if(tool.isEmpty()||!Json.bool(rules,"enabled",true))return new Result("unobserved","没有工具观测或已关闭工具评分",0,false);
  if(Json.num(tool,"hardness",0)<0)return new Result("unbreakable","目标不可挖掘",Json.num(rules,"wrong_tool_amount",1),false);
  double speed=Json.num(tool,"speed",1),best=Json.num(tool,"best_speed",speed);
  boolean matches=Json.bool(tool,"family_matches",false);
  if(Json.bool(tool,"requires_correct_tool",false)&&!Json.bool(tool,"can_harvest",false)){
   return matches?new Result("insufficient_tier","工具等级不足，无法获得掉落物",Json.num(rules,"insufficient_tier_amount",1.5),false):new Result("wrong_tool","工具类型不适合，无法获得掉落物",Json.num(rules,"wrong_tool_amount",1),false);
  }
  if(!matches&&Json.bool(tool,"held_is_tool",false)&&!Json.str(tool,"recommended_family","special_or_hand").equals("special_or_hand"))return new Result("wrong_tool","手持工具不适合这个方块",Json.num(rules,"wrong_tool_amount",1),false);
  if(best>1&&speed/best<Json.num(rules,"minimum_speed_ratio",.5))return new Result("better_tool_available","背包有明显更合适的工具："+Json.str(tool,"best_item",""),Json.num(rules,"inefficient_amount",.25),false);
  if(matches||Json.bool(tool,"requires_correct_tool",false))return new Result("appropriate","工具适合目标方块",0,true);
  return new Result("neutral","可徒手采集，当前没有明显更合适的工具",0,false);
 }
}
