"""Submission PDFs and scripts generated exclusively from actual results."""
import json
import html
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from src.data import ROOT

TITLE='Training-Free Reliability Screening for Prompted Segmentation'
REFS=[
'[1] A. Kirillov et al. Segment Anything. ICCV, 2023. https://arxiv.org/abs/2304.02643',
'[2] J. Quesada, M. Alotaibi, M. Prabhushankar and G. AlRegib. PointPrompt: A Multi-modal Prompting Dataset for Segment Anything Model. CVPR Workshops, 2024. https://openaccess.thecvf.com/content/CVPR2024W/PV/html/Quesada_PointPrompt_A_Multi-modal_Prompting_Dataset_for_Segment_Anything_Model_CVPRW_2024_paper.html',
'[3] T. Kaiser, T. Norrenbrock and B. Rosenhahn. UncertainSAM: Fast and Efficient Uncertainty Quantification of the Segment Anything Model. ICML, PMLR 267, pp. 28670-28688, 2025. https://proceedings.mlr.press/v267/kaiser25a.html',
'[4] S. Loke et al. When Does Prompt-Perturbation Uncertainty Catch Interactive-Segmentation Failures? MICCAI satellite open-access paper, UNSURE, 2026. https://papers.miccai.org/miccai-2026-sat/UNSURE2026_002.html',
'[5] V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman. Geodesic Star Convexity for Interactive Image Segmentation. CVPR, 2010. Dataset: https://robots.ox.ac.uk/~vgg/data/iseg/',
'[6] O. M. Parkhi, A. Vedaldi, A. Zisserman and C. V. Jawahar. Cats and Dogs. CVPR, 2012. Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/'
]
NAMES={'confidence':'SAM confidence','stability':'Threshold stability','consistency':'Mean prompt consistency','minimum':'Minimum consistency','fusion':'Confidence + consistency','flip':'Horizontal-flip consistency'}
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyResearch',fontName='Helvetica',fontSize=9.3,leading=12.5,spaceAfter=7))
styles.add(ParagraphStyle(name='CaptionResearch',fontName='Helvetica',fontSize=8,leading=10.5,spaceAfter=8,textColor=colors.HexColor('#425466')))
styles.add(ParagraphStyle(name='TitleResearch',fontName='Helvetica-Bold',fontSize=17,leading=20,spaceAfter=12,textColor=colors.HexColor('#16324f')))
styles.add(ParagraphStyle(name='SectionResearch',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=5,spaceAfter=8,textColor=colors.HexColor('#16324f')))

def p(text,style='BodyResearch'):return Paragraph(html.escape(text),styles[style])
def heading(text):return p(text,'SectionResearch')
def picture(name,width=500,height=None):
    from PIL import Image as PILImage
    path=ROOT/'figures'/name;w,h=PILImage.open(path).size
    return Image(str(path),width=width,height=height or width*h/w)
def number(x):return 'undefined' if x is None else f'{x:.3f}'
def ci_text(result):
    if result['ci95'] is None:return 'undefined interval'
    lo,hi=result['ci95'];return f"{result['estimate']:+.4f} (95% image-bootstrap CI {lo:+.4f} to {hi:+.4f})"
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.gray)
    canvas.drawString(42,25,'Frozen SAM | Controlled simulated clicks | Course research project')
    canvas.drawRightString(A4[0]-42,25,str(doc.page))
def build_pdf(path,story):
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=42).build(list(story),onFirstPage=footer,onLaterPages=footer)
def table(rows,widths):
    values=[[p(str(v),'CaptionResearch') for v in row] for row in rows]
    t=Table(values,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),
                          ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#16324f')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.grey),
                          ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
    return t

def main():
    summary=json.loads((ROOT/'results/summary.json').read_text());boot=json.loads((ROOT/'results/bootstrap.json').read_text());cfg=json.loads((ROOT/'configs/study.json').read_text())
    frozen=json.loads((ROOT/'configs/frozen.json').read_text());main=summary['datasets']['test'];extra=summary['datasets']['extra'];cell=summary['selected']
    if main['cases']!=212 or extra['cases']!=148:raise ValueError('Configured complete experiment required for final materials')
    env=json.loads((ROOT/'results/test.metadata.json').read_text());delta=boot['test']['fusion_aurc']
    direction='lower' if delta['estimate']<0 else 'higher'
    evidence='The interval excludes zero.' if delta['ci95'][0]>0 or delta['ci95'][1]<0 else 'The interval includes zero; the study does not establish a clear improvement.'
    empirical=f"On the main test set, fusion has {direction} AURC than SAM confidence: paired difference {ci_text(delta)}. {evidence}"
    proposal=[p(TITLE,'TitleResearch'),heading('Research question and motivation'),
      p('A plausible segmentation can be wrong while appearing stable and receiving a high model confidence score. We investigate whether small perturbations of a user click provide useful evidence for screening failures of a frozen Segment Anything Model (SAM), and when consistency misses stable errors. The goal is to identify unreliable outputs, not to train a new segmenter or guarantee accurate masks.'),
      heading('Prior work and scope'),p('SAM provides promptable segmentation and predicted mask quality [1]. PointPrompt studies prompting behavior [2]. UncertainSAM learns a lightweight uncertainty estimator [3], whereas this project updates no weights. A recent prompt-perturbation study reports mixed or negative results on medical data [4]. Our contribution is a controlled natural-image study of click placement, perturbation radius, inference budget and transfer to a separately collected public corpus; prompt perturbation itself is not claimed as new.'),
      heading('Data, method and resources'),p('We use 151 images from the Oxford interactive segmentation benchmark, with 45 image-level validation cases and 106 held-out images. An additional internet-downloaded corpus contains 74 Oxford-IIIT Pet test images, two per breed, using official trimaps. Uncertain annotation borders are ignored. For each image, we simulate an interior click and a near-boundary click; these use annotations only to define the target and evaluate output. Official SAM ViT-B weights remain frozen. The scorer compares the original prediction with masks from two, four or eight perturbed clicks at radii of one, three or six percent of the image diagonal. Fixed fusion averages consistency and clipped SAM confidence. A horizontal-flip baseline requires an extra image encoding. Computation uses the verified local device, without any model training.'),
      heading('Evaluation and expected contribution'),p('We compare SAM confidence, threshold stability, mean and minimum prompt consistency, fixed fusion and flip consistency. Validation selective risk selects the primary radius and budget before test evaluation. We report failure-detection AUROC and average precision, selective risk curves, AURC, retained IoU, inference costs and paired image-bootstrap intervals. Failure thresholds are IoU below 0.5 and 0.75. Perturbations leaving the target are analyzed explicitly rather than filtered using labels. We expect interpretable plots, error examples and evidence about whether extra inference is worthwhile; a negative finding is a valid outcome. If useful, the resulting scoring code can flag masks for human review without retraining.'),
      heading('References'),*[p(r,'CaptionResearch') for r in REFS[:4]]]
    build_pdf(ROOT/'submission/proposal.pdf',proposal)
    report=[p(TITLE,'TitleResearch'),p('Course project | SAM ViT-B, frozen weights | Natural-image failure screening','CaptionResearch'),heading('Abstract'),
      p(f"We investigate whether prompt-response consistency identifies failures of a frozen Segment Anything Model without training an uncertainty estimator. A validation-only choice of perturbation radius and budget is evaluated on {main['images']} held-out natural images ({main['cases']} simulated click cases) and {extra['images']} separately downloaded Pet images ({extra['cases']} cases). Baselines include predicted IoU, threshold stability and horizontal-flip consistency; mean and minimum prompt consistency and a fixed confidence-consistency fusion are evaluated. The selected configuration is {cell}. Main-set mean segmentation IoU is {main['mean_iou']:.3f}. {empirical} We find {main['stable_errors']} stable failures under the predefined consistency >= 0.9 and IoU < 0.5 rule. The contribution is a reproducible empirical analysis with image-level uncertainty estimates, not a new backbone or a claim that consistency guarantees correctness."),
      heading('1. Introduction'),p('Interactive segmentation reduces annotation effort by letting a user specify a target with a click. However, the resulting mask may cover an object part, include nearby background, or select another plausible structure. A quality score is useful only if it ranks such failures. Repeating a segmentation under nearby prompts offers a simple test: do small prompt changes cause large output changes? This hypothesis is attractive because SAM can reuse one image embedding for many mask decodes. It also has a strong counterargument: a model can repeat the same mistake, and a perturbed click can accidentally specify a different object.'),
      p('We therefore ask whether consistency helps beyond the model self-rating, how the result depends on click location and perturbation scale, and whether the added inference cost is justified. All methods score the same original mask. We keep failures and null findings in the analysis. This is a controlled simulated-click study rather than an evaluation of human interaction.'),
      heading('2. Related work'),p('SAM [1] supplies pretrained segmentation and a predicted IoU head. PointPrompt [2] motivates studying prompting behavior, without implying that more prompts are always beneficial. UncertainSAM [3] trains a post-hoc uncertainty model and is outside our strict no-training protocol. Prompt-perturbation uncertainty has already been studied directly [4], including findings that differences may not survive paired comparisons. Our study independently implements a label-blind scorer and examines natural-image cases, budgets, radii, simulated click regimes and an additional Pet corpus. We do not claim to originate perturbation-based uncertainty.'),
      PageBreak(),heading('3. Data and methodology'),
      p('The main Oxford interactive segmentation corpus [5] contains 151 annotated images originating from GrabCut, PASCAL VOC and Alpha matting. A seeded shuffle assigns 45 images to validation and 106 to test, before generating any clicks. The extra corpus selects two official Pet test images per breed [6], for 74 images. Image identities, archive sources, hashes and preparation rules are saved. Extra images are internet-collected public data, not personally photographed images. Exact decoded-image duplicate checks prevent accidental overlap in our evaluated corpora; they cannot establish absence from SAM pretraining.'),
      p('Main annotations map 255 to foreground, 0 to background and 128 to an ignored border. Pet trimaps map 1 to foreground, 2 to background and 3 to ignored border. Original image coordinates and dimensions are retained for evaluation. The foreground distance-transform maximum simulates an interior click. A seeded positive pixel within 2% of the image diagonal from the object boundary simulates a boundary click, with a logged interior fallback if that band is empty. Annotation-dependent simulation and diagnostics are separate from inference and scoring.'),
      p('For each click, SAM generates three candidates; highest predicted IoU chooses the original prediction. The selection never uses annotation overlap. Eight fixed angular offsets define perturbations at radii 1%, 3% and 6% of the image diagonal. K=2 uses opposite offsets and K=4 uses cardinal offsets; K=8 uses all directions. Coordinates are clipped to the image. Perturbations are not repaired using reference masks. The fraction leaving the target is recorded for evaluation only.'),
      heading('3.1 Reliability scores'),table([
       ['Method','Higher-is-better score','Additional inference'],
       ['SAM confidence','Predicted IoU head','None'],['Threshold stability','Overlap at logits threshold +/- 1','None'],
       ['Mean / minimum consistency','Mean / minimum IoU(original, jitter mask)','K cached-embedding decodes'],
       ['Fixed fusion','(clipped confidence + mean consistency) / 2','K cached-embedding decodes'],
       ['Flip consistency','IoU(original, inverse-flipped prediction)','One encoding and one decode']], [130,245,135]),
      p('All reliability functions are annotation-free. The original prediction is never changed. Confidence and fusion are ranking scores, not calibrated correctness probabilities. Empty-empty agreement is 1; an empty prediction against a valid nonempty target has true IoU 0. Empty threshold-stability denominator is assigned reliability 0, explicitly avoiding undefined division.'),
      heading('3.2 Evaluation protocol'),p('Selective risk is mean (1 - true IoU) among retained cases, sorted by reliability. AURC averages that risk at each coverage k/n. Tied score groups use expected loss under random ordering within the group. Failure AUROC and average precision use risk = 1 - reliability; one-class subsets are undefined. Radius and K minimize validation fusion AURC, with lower K then radius as tie-breakers. Paired bootstrap resamples image IDs 2,000 times, keeping both prompt regimes together. Subgroup analyses are exploratory.'),
      PageBreak(),heading('4. Main held-out results'),p(f"Validation selected {cell} before held-out evaluation. The main test includes {main['images']} images and {main['cases']} cases; {main['methods']['confidence']['failure_0.5']['failures']} cases have IoU < 0.5. Full-coverage mean IoU is {main['mean_iou']:.3f} and is identical across reliability methods, because masks are unchanged. Lower AURC is better; higher failure AUROC and average precision are better."),
      table([['Method','AURC','AUROC < .5','AP < .5','IoU @ 75%']]+[[NAMES[m],number(v['aurc']),number(v['failure_0.5']['auroc']),number(v['failure_0.5']['average_precision']),number(v['retained_iou']['0.75'])] for m,v in main['methods'].items()],[195,65,80,70,100]),
      Spacer(1,8),picture('test_risk.png',500),p('Figure 1. Main-test selective risk. The random line is expected performance without quality ranking; the oracle curve uses annotations and is not a deployable method. All practical methods rank the same masks.','CaptionResearch'),
      p(empirical),p(f"Mean consistency alone has observed AURC {main['methods']['consistency']['aurc']:.3f}, compared with fusion {main['methods']['fusion']['aurc']:.3f}. Thus the study supports consistency-based screening beyond SAM confidence, but does not demonstrate an additional benefit from fusion over consistency alone."),p('Other paired comparisons against confidence, with image-cluster bootstrap: prompt-consistency AURC difference '+ci_text(boot['test']['consistency_aurc'])+'; flip-consistency AURC difference '+ci_text(boot['test']['flip_aurc'])+'. AUROC differences and secondary IoU < 0.75 metrics are saved with the full results. These intervals quantify sample uncertainty within this protocol, not all deployment settings.'),
      PageBreak(),heading('4.1 Sensitivity and additional corpus'),picture('sensitivity.png',500),
      p('Figure 2. Registered radius/budget sensitivity for fixed fusion. These held-out cells are descriptive; the primary configuration remains the validation choice, even if another test cell appears better.','CaptionResearch'),
      p(f"The Pet corpus contains {extra['images']} images and {extra['cases']} cases, with mean IoU {extra['mean_iou']:.3f}. SAM confidence AURC is {extra['methods']['confidence']['aurc']:.3f}; fusion AURC is {extra['methods']['fusion']['aurc']:.3f}. The paired difference is {ci_text(boot['extra']['fusion_aurc'])}. This is a second public corpus and does not prove generalization to unseen pretraining images."),
      picture('extra_risk.png',430),p('Figure 3. Additional Pet-corpus selective risk, with the same frozen configuration and score definitions.','CaptionResearch'),
      p(f"The selected perturbations leave the annotated target in an average fraction {main['mean_outside_fraction']:.3f} on the main set and {extra['mean_outside_fraction']:.3f} on Pet. The all-inside diagnostic retains {main['all_perturbations_inside']['cases']} main cases and {extra['all_perturbations_inside']['cases']} Pet cases. It helps distinguish sensitivity to the prompt from accidentally changing the target, but uses labels and is not part of a deployable scorer. Results by click regime, object size and confidence are supplied in summary.json."),
      p(f"Within main-test click regimes, confidence/fusion AURC is {main['by_regime']['interior']['confidence']['aurc']:.3f}/{main['by_regime']['interior']['fusion']['aurc']:.3f} for interior clicks and {main['by_regime']['boundary']['confidence']['aurc']:.3f}/{main['by_regime']['boundary']['fusion']['aurc']:.3f} for boundary clicks. In the all-inside diagnostic, main confidence/fusion AURC is {main['all_perturbations_inside']['methods']['confidence']['aurc']:.3f}/{main['all_perturbations_inside']['methods']['fusion']['aurc']:.3f}. The observed improvement therefore persists descriptively within both click regimes and when no perturbation leaves the target; these restricted analyses do not establish a causal mechanism."),
      PageBreak(),heading('5. Failure analysis and discussion'),picture('examples.png',480),
      p('Figure 4. Actual held-out predictions, annotated references and simulated clicks. Rows include stable failures when available, additional low-IoU cases and a successful case. Numerical captions show true IoU, SAM self-rating and prompt consistency.','CaptionResearch'),
      p(f"Under the predefined rule, {main['stable_errors']} main cases and {extra['stable_errors']} Pet cases are stable errors. Agreement measures repeatability, not semantic correctness. A mask may consistently identify a part rather than the annotated whole object. Conversely, a correct mask can be sensitive near an ambiguous boundary. Finite sample size, only one backbone, deterministic direction probes and annotation-defined click placement limit generalization. Ignored borders improve label validity but leave boundary quality only partially evaluated."),
      p('The full sweep computes more predictions than a deployment using the selected K. Five validation images separately benchmark true K=2,4,8 batches and flip inference; measured timings and the cost plot are included in the package. GPU timing is synchronized. Perturbation scoring reuses one embedding; flip consistency re-encodes the image. Any cost advantage must be interpreted together with ranking accuracy, not as evidence of better masks.'),
      heading('6. Conclusion'),p(empirical+' The results answer a bounded research question about reliability ranking under simulated prompts. Further work should test human clicks, additional backbones and deployment shifts before treating any reliability score as a decision guarantee.'),
      heading('Individual contributions'),p('Real team identities and actual contributions have not been provided. Each student must replace the pending entries in team.json with their name, student ID and verified work, and record their own presentation segment. Automated implementation does not certify the required individual work hours.'),
      PageBreak(),heading('References'),*[p(r) for r in REFS],heading('Reproducibility'),
      p(f"Seed: {cfg['seed']}. Frozen selection: {cell}. Device: {env['device']}; PyTorch {env['torch_version']}. Model SHA-256: {env['model_hash']}. Raw sources, dependency versions, splits, per-case masks, validation selection and bootstrap output are saved in the project. No parameters were trained or updated."),
      p('The assignment gives inconsistent statements about proposal grading and an undated October 7 deadline; use the current course announcement for administrative details. The rehearsal video is an aid and does not satisfy the all-student speaking requirement.')]
    build_pdf(ROOT/'submission/report.pdf',report)
    # Editable research prose accompanies the compiled PDF.
    def source(story):
        def plain(v):
            if isinstance(v,Paragraph):return v.getPlainText()
            if isinstance(v,(list,tuple)):return ' '.join(plain(a) for a in v)
            return str(v)
        blocks=[]
        for x in story:
            if isinstance(x,Paragraph):blocks.append(x.getPlainText())
            elif isinstance(x,Table):
                rows=[[plain(v) for v in row] for row in x._cellvalues]
                blocks.append('\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(row)+' |' for row in rows[1:]]))
            elif isinstance(x,Image):blocks.append(f'![Research figure](../figures/{Path(x.filename).name})')
        return '\n\n'.join(blocks)
    (ROOT/'submission/proposal.md').write_text(source(proposal),encoding='utf-8')
    (ROOT/'submission/report.md').write_text(source(report),encoding='utf-8')
    if not (ROOT/'submission/team.json').exists():
        (ROOT/'submission/team.json').write_text(json.dumps({'status':'pending real student information','members':[],
                                'required_fields':['name','student_id','actual_contribution','actual_hours','presentation_segment']},indent=2))
    narratives=[
      ('Research question',f"Our project asks when a frozen segmentation model can recognize its own failures. We use the original Segment Anything Model, or SAM, without training or fine-tuning. The aim is to rank unreliable masks for review. We do not alter the mask itself. Our strongest counterargument is simple: a model can produce the same wrong segmentation repeatedly. We test that possibility instead of assuming stability means correctness."),
      ('Related work','SAM predicts its own mask quality. PointPrompt studies the role of prompts. Prompt perturbation uncertainty also already exists, including a recent medical-image study with largely negative comparisons. We therefore do not claim a new uncertainty principle. Our contribution is a controlled natural-image analysis of click position, perturbation radius, compute budget, and a separate public image corpus.'),
      ('Protocol',f"We split the main dataset by image: forty-five validation images and one hundred and six test images. We separately download seventy-four Pet images, two per breed. Each image has an interior and a boundary click simulated from annotations. Labels are used only for simulation and evaluation. All scoring code receives the image, click and predictions. Uncertain annotation borders are ignored, and every image's two clicks stay together in statistical resampling."),
      ('Methods',f"The baselines are SAM confidence, threshold stability and horizontal-flip consistency. Our prompt score averages overlap between the original mask and nearby-click masks. We also test minimum overlap and equal-weight fusion with confidence. We sweep three radii and three budgets. Validation chooses {cell}; that choice is frozen before test evaluation. Nearby clicks may leave the object, so we record that diagnostic instead of secretly fixing prompts with labels."),
      ('Main results',f"On the main test set the original masks have mean IoU {main['mean_iou']:.3f}. Confidence has AURC {main['methods']['confidence']['aurc']:.3f}, and fusion has AURC {main['methods']['fusion']['aurc']:.3f}. Lower is better. The paired difference is {delta['estimate']:+.4f}. The ninety-five percent interval runs from {delta['ci95'][0]:+.4f} to {delta['ci95'][1]:+.4f}. {'It includes zero, so a clear improvement is not established.' if delta['ci95'][0]<=0<=delta['ci95'][1] else 'It excludes zero within this evaluation protocol.'} The curves show whether refusing low-confidence cases actually lowers retained error."),
      ('Additional corpus',f"We apply exactly the same configuration to the Pet corpus. Mean IoU is {extra['mean_iou']:.3f}; confidence AURC is {extra['methods']['confidence']['aurc']:.3f}, compared with fusion AURC {extra['methods']['fusion']['aurc']:.3f}. This checks sensitivity to a second set of natural images. It does not establish that those images were absent from SAM pretraining. We separately inspect radius and budget sensitivity, and report subgroup results as exploratory."),
      ('Stable failures',f"We define a stable failure in advance as prompt consistency at least point nine and true IoU below point five. There are {main['stable_errors']} such main-test cases and {extra['stable_errors']} Pet cases. The examples show the original image, the reference object and the prediction. These cases reveal the fundamental limitation: agreement can describe a repeated mistake. A reliable screening method must be tested against labels, even when it does not use labels at inference."),
      ('Compute','Prompt variations reuse one cached image embedding, while horizontal flip needs another encoder pass. We measure actual batches of two, four and eight decodes on five validation images, rather than assuming cost scales linearly. The full experimental sweep costs more than deploying just the chosen method. Quality and cost must be read together: cheap inference is useful only if its ranking is informative.'),
      ('Conclusion','The project provides executable code, preserved observations, controlled comparisons and image-level bootstrap intervals, with no training. Its conclusion follows the measured results, including negative findings. Limits include simulated users, one backbone and finite datasets. Future work should test real user clicks and additional models. Each team member should present their assigned section and document their actual contribution.')]
    script='\n\n'.join(f'## Slide {i+1}: {title}\n\n{text}' for i,(title,text) in enumerate(narratives))
    (ROOT/'submission/narration_en.md').write_text('# Presentation script\n\nTarget: 4-4.5 minutes. Rehearse and adjust speaking pace. Allocate all sections among actual students; every member must speak.\n\n'+script,encoding='utf-8')
    zh='''# 中文理解与录制说明

研究问题：不训练模型，利用点击扰动后的分割一致性，能否比 SAM 自带评分更好地发现错误？稳定不等于正确，因此重点检查“稳定地画错”。

第1页：解释任务是可靠性排序，不修改原分割结果。第2页：提示扰动已有研究，项目贡献是独立的自然图像、多因素分析。第3页：45张验证、106张主测试、74张额外Pet图片，每张模拟两类点击；真值只用于模拟和评价。第4页：对比模型自信度、阈值稳定性、均值/最差扰动一致性、固定融合与翻转一致性；只用验证集选参数。第5页：按实际曲线、表格和区间讲主测试结果，区间包含零就不能说显著提升。第6页：讲额外数据与敏感性分析，不能声称这些图片未被SAM预训练见过。第7页：展示稳定但错误的真实案例。第8页：解释缓存图像编码为什么节省额外推理，并按实际计时讲成本。第9页：总结实验真正支持的结论及限制。

英文逐页讲稿在 narration_en.md。所有数字以 report.pdf、results/summary.json 与 bootstrap.json 为准。组员人数尚未知，确认后均分讲稿；必须每人本人发言。预览MP4只是计时与换页辅助，不满足本人讲解要求。目标4至4.5分钟，最多5分钟。
'''
    translations=[
      ('研究问题','我们研究的是：冻结的分割模型什么时候能识别自身的错误？使用原版 SAM，不训练、不微调。目标是把不可靠的分割结果排到前面供人复核，不改变原来的掩码。最强的反对理由是，模型可能反复给出同一个错误分割。因此我们实际检验这个问题，而不假设稳定就意味着正确。'),
      ('相关研究','SAM 自带掩码质量预测。PointPrompt 研究提示行为。提示扰动不确定性也已有直接研究，包括近期一项比较结果主要为负面的医学图像研究。我们不声称发明新的不确定性原理。贡献是对自然图像开展受控分析，检查点击位置、扰动半径、推理预算，以及另一个公开图像集上的表现。'),
      ('实验协议','主数据按图片划分：45 张验证图，106 张测试图。另行下载74张 Pet 图片，每个品种两张。每张图模拟内部点击和边界点击。标签只用于生成模拟点击与评价，评分代码只接收预测。评价忽略标注不确定的边缘；统计重采样时，同一图片的两个点击始终一起抽取。'),
      ('方法',f'基线包括 SAM 置信度、阈值稳定性和水平翻转一致性。提示一致性计算原掩码与附近点击产生的掩码之间的平均重叠；同时比较最小重叠，以及与置信度等权融合。扫描三种半径和三种解码预算，仅验证集选择 {cell}，随后固定。附近点击可能移出目标，因此记录这个诊断量，不使用标签偷偷修正提示。'),
      ('主测试结果',f"主测试原始掩码的平均 IoU 为 {main['mean_iou']:.3f}。置信度 AURC 为 {main['methods']['confidence']['aurc']:.3f}，融合为 {main['methods']['fusion']['aurc']:.3f}；越低越好。成对差值为 {delta['estimate']:+.4f}，95% 区间从 {delta['ci95'][0]:+.4f} 到 {delta['ci95'][1]:+.4f}。"+('区间包含零，不能据此确认明确改善。' if delta['ci95'][0]<=0<=delta['ci95'][1] else '区间在本实验协议下排除了零。')+'风险曲线检验的是：拒绝低评分结果是否真的降低留下结果的错误。'),
      ('额外数据',f"使用同样的固定参数评价 Pet 数据。平均 IoU 为 {extra['mean_iou']:.3f}；置信度 AURC 为 {extra['methods']['confidence']['aurc']:.3f}，融合为 {extra['methods']['fusion']['aurc']:.3f}。这检查另一批自然图像上的表现，不证明这些图片没有进入 SAM 的预训练数据。半径、预算和子组分析为描述性分析，不据此重新选择主参数。"),
      ('稳定错误',f"预先定义一致性至少0.9且真实 IoU 低于0.5为稳定错误。主测试发现 {main['stable_errors']} 个，Pet 数据发现 {extra['stable_errors']} 个。实例并列展示输入、参考目标和预测。这说明一致性可以反映重复的错误。评分本身不使用标签，但检验评分是否可靠必须对照标签。"),
      ('计算成本','提示变化复用缓存的图像编码，水平翻转则需要额外编码一次。在五张验证图上实际测量两次、四次和八次解码批次，不假设成本线性变化。整个研究扫描比部署单个选定方法开销更大。计算成本必须结合排序质量解读，推理便宜并不等于评分有用。'),
      ('结论','项目提供可执行代码、保留的预测记录、受控比较和按图片重采样的统计区间，全程没有训练。结论服从测量结果，包括负面结果。局限包括模拟用户、单一模型和有限的公开数据。后续应评价真实点击与更多模型。实际组员需要分配讲解段落并记录本人真实贡献。')]
    zh+='\n\n'+ '\n\n'.join(f'## 第{i+1}页：{title}\n\n{text}' for i,(title,text) in enumerate(translations))
    (ROOT/'submission/narration_zh.md').write_text(zh,encoding='utf-8')
    (ROOT/'submission/slides_content.json').write_text(json.dumps({'title':TITLE,'slides':[{'title':t,'notes':n} for t,n in narratives]},indent=2))
    print('Created PDFs, editable prose, slide content and narration')

if __name__=='__main__':main()
