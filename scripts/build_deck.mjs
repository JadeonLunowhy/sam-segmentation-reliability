import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime='C:/Users/31479/.cache/codex-runtimes/codex-primary-runtime/dependencies';
const skill='C:/Users/31479/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations';
process.env.RUNTIME_NODE_MODULES=path.join(runtime,'node/node_modules');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {resolvePresentationFont,applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const family=resolvePresentationFont({fontFamily:'Arial'});
const summary=JSON.parse(await fs.readFile(path.join(root,'results/summary.json'),'utf8'));
const boot=JSON.parse(await fs.readFile(path.join(root,'results/bootstrap.json'),'utf8'));
const timing=JSON.parse(await fs.readFile(path.join(root,'results/budget_timing.json'),'utf8'));
const content=JSON.parse(await fs.readFile(path.join(root,'submission/slides_content.json'),'utf8'));
const main=summary.datasets.test,extra=summary.datasets.extra;
const delta=boot.test.fusion_aurc;
const build=path.join(root,'tmp/deck');await fs.mkdir(build,{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});

function text(slide,value,x,y,w,h,size=28,bold=false,color='#18324C') {
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:family,fontSize:size,bold,color,autoFit:'none'};return s;
}
function slide(title,index){
 const s=deck.slides.add();s.background.fill='#FFFFFF';
 text(s,title,64,44,1152,82,44,true);
 text(s,String(index),1180,676,36,22,18,false,'#718096');
 s.speakerNotes.textFrame.setText(content.slides[index-1].notes+'\nSources: docs/design.md; results/summary.json; results/bootstrap.json; results/budget_timing.json.');
 return s;
}
async function image(s,name,x,y,w,h){s.images.add({blob:new Uint8Array(await fs.readFile(path.join(root,'figures',name))),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function nativeTable(s,values,widths,x=64,y=165,w=1152,h=390,size=25){
 const table=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
   const cell=table.getCell(r,c);cell.text.style={typeface:family,fontSize:size,color:'#18324C',bold:r===0,autoFit:'none'};
   cell.fill=r===0?'#E8EFF5':'#FFFFFF';
 }
 return table;
}
const fmt=x=>x===null?'n/a':x.toFixed(3);
const names={confidence:'SAM confidence',stability:'Threshold stability',consistency:'Mean consistency',minimum:'Minimum consistency',fusion:'Fixed fusion',flip:'Flip consistency'};

let s=slide('Training-Free Reliability Screening',1);
text(s,'for Prompted Segmentation',64,133,1150,82,48,true);
text(s,'When does a stable mask still fail?',64,255,1150,70,36);
text(s,'Frozen SAM ViT-B\nControlled natural-image experiments\nNo model training or learned uncertainty head',64,360,1150,210,30);
text(s,'Course project | Real observations and image-level bootstrap intervals',64,620,1150,35,22,false,'#61758B');

s=slide('Research context and contribution',2);
text(s,'Existing foundations',64,155,520,55,32,true);
text(s,'SAM: promptable masks and predicted IoU\nPointPrompt: prompting behavior\nPrompt uncertainty: already studied',64,235,550,220,28);
text(s,'This study',680,155,535,55,32,true);
text(s,'Natural images and an additional Pet corpus\nInterior versus boundary clicks\nRadius, decode budget and stable errors',680,235,535,220,28);
text(s,'The contribution is a controlled analysis; perturbation uncertainty is not claimed as new',64,555,1140,80,28,true);

s=slide('Image-level experimental protocol',3);
nativeTable(s,[['Split','Images','Simulated click cases'],['Validation','45','90'],['Main test',String(main.images),String(main.cases)],['Additional Pet',String(extra.images),String(extra.cases)]],[500,260,392],64,165,1152,300,29);
text(s,'Two click regimes: foreground interior and near boundary',64,500,1150,55,30);
text(s,'Annotations simulate clicks and evaluate masks; reliability scoring never receives them',64,565,1140,85,27);

s=slide('Reliability scores and inference budget',4);
nativeTable(s,[['Score','Additional work'],['SAM confidence / threshold stability','No additional model call'],['Mean / minimum prompt consistency','K decodes; reuse image embedding'],['Fixed fusion: (confidence + consistency) / 2','K decodes; no learned weights'],['Horizontal-flip consistency','One image encoding + one decode']],[715,437],64,155,1152,340,25);
text(s,'Radius: 1%, 3%, 6% of image diagonal | K: 2, 4, 8',64,525,1150,55,28);
text(s,'Primary configuration selected on validation only: '+summary.selected,64,590,1140,55,28,true);

s=slide('Main held-out selective risk',5);
await image(s,'test_risk.png',40,145,795,455);
text(s,'Lower AURC is better',870,180,340,65,29,true);
text(s,`Confidence: ${fmt(main.methods.confidence.aurc)}\nFixed fusion: ${fmt(main.methods.fusion.aurc)}\n\nPaired difference: ${delta.estimate.toFixed(4)}\n95% CI: ${delta.ci95.map(x=>x.toFixed(4)).join(' to ')}`,870,260,340,180,25);
text(s,`Mean consistency: ${fmt(main.methods.consistency.aurc)}\nFusion adds no demonstrated gain over it`,870,465,340,95,22);
text(s,delta.ci95[0]<=0&&delta.ci95[1]>=0?'The interval includes zero; a clear gain is not established':'The paired interval excludes zero within this protocol',64,620,1140,40,25,true);

s=slide('Additional corpus and sensitivity',6);
await image(s,'sensitivity.png',40,145,780,370);
text(s,`Pet: ${extra.images} images / ${extra.cases} cases\n\nConfidence AURC ${fmt(extra.methods.confidence.aurc)}\nFusion AURC ${fmt(extra.methods.fusion.aurc)}\n\nSame frozen configuration`,850,190,365,320,27);
text(s,'Sensitivity cells are descriptive; they do not replace validation selection',64,550,1150,80,28,true);

s=slide('Stable errors in actual predictions',7);
await image(s,'examples_slides.png',64,140,1152,435);
text(s,`Consistency >= 0.9 and true IoU < 0.5: ${main.stable_errors} main cases; ${extra.stable_errors} Pet cases`,64,595,1150,50,25,true);

s=slide('Measured additional inference cost',8);
const mean=fn=>timing.samples.reduce((a,r)=>a+fn(r),0)/timing.samples.length;
const chart=s.charts.add('bar',{position:{left:80,top:155,width:1100,height:400},title:'Mean additional inference time (milliseconds)',categories:['K = 2','K = 4','K = 8','Horizontal flip'],
 series:[{name:'Milliseconds',values:[2,4,8].map(k=>mean(r=>r.additional_prompt_seconds[String(k)])).concat([mean(r=>r.additional_flip_seconds)]).map(v=>Number((v*1000).toFixed(3))),fill:'#28658F'}],
 barOptions:{direction:'column',grouping:'clustered'},hasLegend:false,dataLabels:{showValue:true,position:'outEnd'},yAxis:{numberFormatCode:'0.00'}});
applyPresentationChartFont(chart,{fontFamily:family});
text(s,'Actual batches on five validation images | '+timing.device+' | synchronized GPU timing',64,580,1150,50,25);
text(s,'The complete research sweep costs more than deploying one selected score',64,635,1140,35,24,true);

s=slide('Findings and limitations',9);
const conclusion=delta.estimate<0?'Fusion lowers observed main-test AURC':'Fusion does not lower observed main-test AURC';
text(s,conclusion,64,170,1150,70,34,true);
text(s,'Agreement measures repeatability, not semantic correctness\n\nLimits: simulated users, one backbone, finite public corpora\n\nAll methods leave the original segmentation unchanged',64,280,1140,230,30);
text(s,'Code, raw observations, masks and paired intervals support every reported result',64,575,1140,70,28,true);

const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
for(let i=0;i<deck.slides.items.length;i++){
 const slide=deck.slides.items[i];const png=await deck.export({slide,format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const final=path.join(root,'submission/presentation.pptx');
const validated=path.join(build,`validated-${Date.now()}.pptx`);
const receipt=path.join(root,`tmp/presentation-validation-${Date.now()}.json`);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:validated,
 pythonExecutable:path.join(runtime,'python/python.exe'),integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','4'],
 requiredNativeTableOwnerSlides:[3,4],requiredNativeChartOwnerSlides:[8],fontPolicy:{basis:'design',families:[family]},
 materializeLiteralChartWorkbooks:true,
 verifyArtifactToolImport:true,receiptPath:receipt});
// Preserve the current deliverable until its replacement has passed every gate.
await fs.copyFile(validated,final);
await fs.copyFile(receipt,path.join(root,'results/presentation_validation.json'));
console.log('Created',final);
