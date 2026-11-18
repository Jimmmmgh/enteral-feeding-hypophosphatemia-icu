# 阶段 4.1 新增字段可行性与检查点 3

**通过有限范围。** 原三库筛选流程全部计数复现。数据仅在内存按 ICU 停留链接，留存特征逐列、逐行顺序核对。三库基线资格人数为 6,027、1,086、1,315，主分析人数为 5,622、879、1,233。下表分母为各库主分析人群。带 * 是小于 5 的计数。阳性记录占比衡量表内可捕捉的记录，缺少记录的临床状态未知。

| 字段 | MIMIC-IV | eICU-CRD | SICdb |
|---|---|---|---|
| 开始在基线窗／结局窗的补磷记录 | ICU inputevents 项 225834、225835、225925（静脉），eMAR 给药事件加 eMAR detail 的口服／肠管途径。基线窗任一路径 345/5,622（静脉 281，口服／肠管 66），结局窗 711/5,622（静脉 606，口服／肠管 123）。两路可重叠。 | medication 未取消磷盐医嘱。基线任一路径 72/879（静脉 34，口服／肠管 51），结局窗 132/879（静脉 59，口服／肠管 92，途径不明 1*）。窗口内 791/879 有任意药物医嘱，基线窗 752/879。有任意输注表记录 735/879，输注表内映射到补磷的记录 0*。给药无法用该医嘱变量直接确认。 | 在 medication 以候选磷盐／输注字典 ID 1760、2084、2138 搜索正剂量记录，两个窗内均为 0*。主分析人群全病例期间仅两条候选记录，状态编码临床含义未确认。无法估计实际补磷覆盖率或补磷率。 |
| 索引时 RRT | procedureevents 中间歇血透 225441 及 CRRT/变式 225802、225803、225809、225955 的记录区间覆盖索引时间 312/5,622。derived.rrt 索引前 6h 有活动记录 362/5,622，是另一观察口径。 | treatment 仅有录入时间，索引前 24h 任意治疗记录 495/879、目标血透／连续滤过记录 47/879，索引时状态无有效时间区间。APACHE dialysis 项的 0 可代表未填写。 | cases.HoursOfCRRT 1,233/1,233 非空，280/1,233 >0，指全病例累计。data_float_h 字典有 CRRT 血流、透析液和置换流量观察，不能由信号缺失推定索引时 CRRT 和间歇血透联合阴性状态。 |
| 病情评分 | derived.sofa 的末次索引前 2h 内结束的 24h SOFA 有效值 5,619/5,622。 | apachepatientresult 中 APACHE IV 非负且同一住院一致，782/879 有效。首日 APACHE IV 与索引前 24h SOFA 数据窗和构成不同。 | cases.saps3 正值 1,233/1,233。SAPS III 与 SOFA 非同一测量，缺实际评分完成时点。 |
| 基线窗胰岛素输注记录代理 | inputevents 223258 连续滴注类，正速率及正量并与基线窗重叠，340/5,622。 | infusiondrug 阳性数值速率胰岛素记录，215/879。 | medication 1557 非单剂且正量、记录区间重叠，496/1,233，GivenState 的实质含义待确认。 |
| 窗口终止时间／状态资料 | ICU outtime 5,622/5,622。用 admissions 的 deathtime 和 hospital_expire_flag 判别死亡或离开单元，信息有效 5,621/5,622。 | ICU unitdischargeoffset 和 unitdischargestatus（Alive/Expired）均覆盖 879/879。 | TimeOfStay 病例结案时间 1,233/1,233。DischargeState 有明确生存状态 1,148/1,233，并有 OffsetOfDeath 可关联。结案可以包含多个单元；与实际出 ICU 时间不同。 |
| 入院诊断 | 可识别在索引前完成的 sepsis3 定义，医院 DKA ICD 无记录入院诊断时间。两项均无法直接归为入院诊断。 | 入院诊断路径和索引前 diagnosis 可用于库内疾病记录分组。主分析有入院路径 871/879，任意索引前诊断 859/879。 | ICD10Main 非空 1,232/1,233。另有 AdmissionFormHasSepsis 和 SurgicalSite，分别记录入院脓毒症字段和手术部位。无共同的正式入院分类。 |

eICU 原源资格集 2,376 人、51 家医院中，有合格基线磷 1,342 人。年龄与医院覆盖 2,376/2,376，男性／女性字段 2,375/2,376，索引前入院诊断路径 2,345/2,376，任何索引前诊断 2,288/2,376，非负且一致的 APACHE IV 2,140/2,376。可做 C5 描述性比较。可用性盘点未按低磷结局分组或估计效应。

SICdb 1.0.8 官方网页和版本随附 Documentation.pdf 未定义 GivenState 各码临床含义。当地参考字典只映射 2193、2194、2195、2196 到 State_0、State_3、State_1、State_2。正文应使用 qualifying feeding records。官方字段说明称 WeightOnAdmission 按约 5 kg 粒度取整，未声明文件数值存储为克。主分析样本原始体重四分位数为 70,000、80,000、90,000，除以 1,000 对应 70、80、90 kg。转换依据是发布文件的数值尺度与官方取整陈述，代码沿用原处理。

## 进入 4.2 的范围

- C5、C6 可运行。C6 将 SICdb 非死亡终止称为 MetaVision 结案，单列其中临床出 ICU 状态未知的记录。
- C7 独立报告 MIMIC-IV 给药记录敏感性以及 eICU-CRD 药物医嘱记录敏感性。eICU 仅在基线窗有药物医嘱可见的子集做排除。SICdb 两窗口无足够识别的补磷记录，停做补磷敏感性。
- C8 使用 MIMIC-IV 区间重叠定义，eICU 与 SICdb 的索引时联合 RRT 状态无法核实。C11 跨库设计 HOLD，需同一构成、同一时间窗的病情评分和索引时 RRT。MIMIC-IV 可在追加仅已记录于索引时的 RRT 特征后测试库内可行性。
- 新增代码与聚合覆盖保存在修订目录。无新增受试者级文件。所有原分析文件保留。

官方依据 [SICdb 1.0.8](https://physionet.org/content/sicdb/1.0.8/) · [SICdb 字段说明](https://www.sicdb.com/Documentation/SICdb_Documentation) · [eICU medication](https://eicu.mit.edu/eicutables/medication/) · [eICU treatment](https://eicu.mit.edu/eicutables/treatment/) · [eICU APACHE IV](https://eicu.mit.edu/eicutables/apachepatientresult/) · [MIMIC SOFA](https://github.com/MIT-LCP/mimic-iv/blob/master/concepts/score/sofa.sql)
