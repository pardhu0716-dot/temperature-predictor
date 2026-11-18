"""Build the complete presentation from preserved measurements; never train a model."""
import argparse,csv,gzip,hashlib,json,math,tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate,Frame,PageTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,KeepTogether,Flowable
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--work-dir',type=Path,default=Path(tempfile.mkdtemp(prefix='stage-b-pdf-')));args=p.parse_args();WORK=args.work_dir;WORK.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
core=read(ROOT/'seven_models/original_comparison.json');reg=read(ROOT/'optimizer_regularization/original_results.json');final=read(ROOT/'final_evaluation/evaluation.json')
C=core['results'];R=reg['results'];U=[r for r in R if not r['reused']]
MODEL_NAMES={'ols':'Linear regression','ridge':'Ridge','lasso':'Lasso','elastic_net':'Elastic Net','random_forest':'Random Forest','xgboost':'XGBoost','mlp':'MLP','linear_gradient':'Linear'}
COLORS={'ols':'#4275a6','ridge':'#25a4ba','lasso':'#159263','elastic_net':'#a48a22','random_forest':'#d46536','xgboost':'#9766aa','mlp':'#ce6584','linear_gradient':'#4275a6'}
OPTS=['batch_gd','sgd','adam'];OPT_LABELS=['Batch GD','SGD','Adam'];OPT_COLORS=['#4275a6','#d18132','#875ca8'];PENS=['none','l1','l2','l1_l2'];PEN_LABELS=['None','L1','L2','L1+L2'];PEN_COLORS=['#8b96a3','#159263','#3282b8','#875ca8']
METRICS=[('mae_c','MAE','°C'),('rmse_c','RMSE','°C'),('max_absolute_error_c','Maximum absolute error','°C'),('train_seconds','Training time','s'),('median','Median prediction latency','ms'),('size','Serialized bundle size','bytes')]
def val(r,key):
 if key=='median':return r['latency']['single_sample_median_ms']
 if key=='size':return r['serialized_model_bytes']
 if key=='train_seconds':return r[key]
 return r['validation'][key]
def fmt(v):return f'{v:,.0f}' if v>=10000 else f'{v:.6g}'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':False})
charts=[]
def fig():return plt.subplots(figsize=(7.4,3.25),layout='constrained')
def add(section,title,figure,captions,provenance):
 captions=[s.strip() for s in captions]
 assert len(captions)==2 and all(s.endswith('.') for s in captions)
 i=len(charts)+1;path=WORK/f'figure_{i:02d}.pdf';figure.savefig(path,format='pdf');plt.close(figure)
 charts.append(dict(number=i,section=section,title=title,captions=captions,figure_file=str(path),data=provenance))
def bars(section,title,labels,values,colors,axis,unit,source,horizontal=False,log=False):
 f,a=fig();idx=np.arange(len(labels))
 if horizontal:
  b=a.barh(idx,values,color=colors);a.set_yticks(idx,labels);a.invert_yaxis();a.set_xlabel(axis+' ('+unit+')');a.bar_label(b,labels=[fmt(x) for x in values],padding=4,fontsize=9)
  if log:a.set_xscale('log');a.set_xlim(min(values)*.55,max(values)*6)
  else:a.set_xlim(0,max(values)*1.24)
 else:
  b=a.bar(idx,values,color=colors,width=.7);a.set_xticks(idx,labels);a.set_ylabel(axis+' ('+unit+')');a.bar_label(b,labels=[fmt(x) for x in values],padding=3,fontsize=9)
  if log:a.set_yscale('log');a.set_ylim(min(values)*.4,max(values)*6)
  else:a.set_ylim(0,max(values)*1.25)
 a.grid(axis='x' if horizontal else 'y',alpha=.18);a.set_axisbelow(True)
 best=int(np.argmin(values))
 scale=' on a logarithmic scale' if log else ''
 add(section,title,f,[f'{axis} is compared across the labeled configurations, with values in {unit}{scale}.',f'{labels[best].replace(chr(10)," ")} has the lowest measured value at {fmt(values[best])} {unit}.'],dict(source=source,labels=labels,values=values,unit=unit))
# Six original family comparisons, using the original timing snapshot.
for key,label,unit in METRICS:
 bars('Seven-model comparison',label+' across seven model families',[MODEL_NAMES[x['experiment']['name']] for x in C],[val(x,key) for x in C],[COLORS[x['experiment']['name']] for x in C],label,unit,'seven_models/original_comparison.json',horizontal=True,log=key in ('train_seconds','median','size'))
# Existing six optimizer entries; consistent optimizer colors.
O=[x for x in R if x['group']=='optimizers']
for key,label,unit in METRICS:
 labels=[('Linear' if x['experiment']['kind']=='linear_gradient' else 'MLP')+'\n'+OPT_LABELS[OPTS.index(x['effective_parameters']['optimizer'])] for x in O]
 bars('Optimizer experiments',label+' for the six optimizer entries',labels,[val(x['report'],key) for x in O],OPT_COLORS*2,label,unit,'optimizer_regularization/original_results.json: optimizers',log=key in ('train_seconds','size'))
# Existing ten regularization entries.
G=[x for x in R if x['group']=='regularization'];GN=[x['experiment']['name'].replace('_',' ') for x in G]
for key,label,unit in [('rmse_c','RMSE','°C'),('max_absolute_error_c','Maximum absolute error','°C')]:
 bars('Regularization experiments',label+' for all ten regularization entries',GN,[val(x['report'],key) for x in G],[COLORS[x['experiment']['kind']] for x in G],label,unit,'optimizer_regularization/original_results.json: regularization',horizontal=True)
# Native OLS / Lasso / Ridge / Elastic Net direct accuracy comparison.
N=[next(x for x in G if x['experiment']['name']==n) for n in ('ols','ridge','lasso','elastic_net')]
f,a=fig();xx=np.arange(4)
for j,(key,lab,_) in enumerate(METRICS[:3]):
 values=[val(x['report'],key) for x in N];b=a.bar(xx+(j-1)*.24,values,width=.23,label=lab);a.bar_label(b,labels=[f'{v:.2f}' for v in values],padding=2,fontsize=9)
a.set_xticks(xx,['OLS','Ridge α=1','Lasso α=.001','Elastic Net']);a.set_ylabel('Validation error (°C)');a.legend(ncols=3,loc='upper center');a.set_ylim(0,12.3)
add('Regularization experiments','Native linear estimators: no penalty, L2, L1 and Elastic Net',f,['Grouped bars compare MAE, RMSE and maximum error in degrees Celsius for the four native linear estimators.','Lasso has the lowest errors, including RMSE 1.463817°C versus Ridge 2.248562°C and Elastic Net 1.613383°C.'],dict(source='regularization group',names=[x['experiment']['name'] for x in N],values=[[val(x['report'],k) for k,_,_ in METRICS[:3]] for x in N]))
for key,label,_ in METRICS[:2]:
 f,a=fig();data=[]
 for names,family in [(('ridge_weak','ridge'),'ridge'),(('lasso_weak','lasso'),'lasso')]:
  es=[next(x for x in G if x['experiment']['name']==n) for n in names];xs=[x['effective_parameters']['alpha'] for x in es];ys=[val(x['report'],key) for x in es]
  a.plot(xs,ys,'o--',color=COLORS[family],label=MODEL_NAMES[family]);data.append(dict(family=family,alpha=xs,values=ys))
  for x,y in zip(xs,ys):a.annotate(f'{y:.4f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
 a.set_xscale('log');a.set_xlabel('Native estimator alpha (log scale)');a.set_ylabel(label+' (°C)');a.legend();a.margins(.2,.25)
 add('Regularization experiments','Two measured alpha settings per native family: '+label,f,[f'The markers show measured validation {label} in degrees Celsius at two tested alpha values for Ridge and Lasso.',f'Lasso at alpha 0.001 has the lowest {label} at {val(next(x for x in G if x["experiment"]["name"]=="lasso")["report"],key):.6f}°C and dashed lines only connect measured points.'],data)
for names,title in [(('mlp_no_penalty','mlp_l2'),'MLP Adam: no penalty versus L2'),(('rf_shallow','random_forest'),'Random Forest: shallow versus default complexity')]:
 es=[next(x for x in G if x['experiment']['name']==n) for n in names];f,a=fig();xx=np.arange(3)
 for j,x in enumerate(es):
  vals=[val(x['report'],k) for k,_,_ in METRICS[:3]];b=a.bar(xx+(j-.5)*.32,vals,width=.3,label=x['experiment']['name'].replace('_',' '));a.bar_label(b,labels=[f'{v:.3f}' for v in vals],padding=3,fontsize=9)
 a.set_xticks(xx,['MAE','RMSE','Maximum error']);a.set_ylabel('Validation error (°C)');a.legend();a.set_ylim(0,max(val(x['report'],'max_absolute_error_c') for x in es)*1.3)
 winner=min(es,key=lambda x:val(x['report'],'rmse_c'))
 add('Regularization experiments',title,f,['Grouped bars compare MAE, RMSE and maximum error in degrees Celsius for the two labeled configurations.',f'{winner["experiment"]["name"].replace("_"," ")} gives lower RMSE at {val(winner["report"],"rmse_c"):.6f}°C.'],dict(names=names,values=[[val(x['report'],k) for k,_,_ in METRICS[:3]] for x in es]))
# All 24 entries, not 24 new independent fits.
B=[x for x in R if x['group']=='combinations']
def combo(kind,opt,pen):return next(x for x in B if x['experiment']['kind']==kind and x['effective_parameters']['optimizer']==opt and x['penalty']==pen)
for kind in ('linear_gradient','mlp'):
 model=MODEL_NAMES[kind]
 for key,label,unit in [METRICS[1],*METRICS[3:]]:
  f,a=fig();xx=np.arange(3);matrix=[]
  for j,pen in enumerate(PENS):
   vals=[val(combo(kind,opt,pen)['report'],key) for opt in OPTS];matrix.append(vals);b=a.bar(xx+(j-1.5)*.19,vals,width=.18,color=PEN_COLORS[j],label=PEN_LABELS[j])
   if key not in ('median','size'):a.bar_label(b,labels=[f'{v:.3g}' for v in vals],padding=2,fontsize=9)
  a.set_xticks(xx,OPT_LABELS);a.set_ylabel(label+' ('+unit+')');a.legend(ncols=4,loc='upper center',fontsize=9);maximum=max(max(x) for x in matrix);a.set_ylim(0,maximum*1.35)
  winner=min([combo(kind,o,p) for o in OPTS for p in PENS],key=lambda x:val(x['report'],key))
  add('Optimizer × regularization matrix',model+': '+label+' by optimizer and penalty',f,[f'Within each {model} optimizer group, four penalties are compared using {label} on the vertical axis in {unit}.',f'{winner["effective_parameters"]["optimizer"]} with {winner["penalty"]} has the lowest measured {label} at {fmt(val(winner["report"],key))} {unit}.'],dict(kind=kind,metric=key,optimizer_order=OPTS,penalty_order=PENS,values_by_penalty=matrix))
 # Swap grouping to show the same penalty under three optimizers.
 f,a=fig();xx=np.arange(4);matrix=[]
 for j,opt in enumerate(OPTS):
  vals=[val(combo(kind,opt,pen)['report'],'rmse_c') for pen in PENS];matrix.append(vals);b=a.bar(xx+(j-1)*.25,vals,width=.24,color=OPT_COLORS[j],label=OPT_LABELS[j]);a.bar_label(b,labels=[f'{v:.3f}' for v in vals],padding=2,fontsize=9)
 a.set_xticks(xx,PEN_LABELS);a.set_ylabel('Validation RMSE (°C)');a.legend(ncols=3);a.set_ylim(0,max(max(x) for x in matrix)*1.3)
 win=min([combo(kind,o,p) for o in OPTS for p in PENS],key=lambda x:val(x['report'],'rmse_c'))
 add('Optimizer × regularization matrix',model+': compare optimizers within each penalty',f,['Each penalty group compares Batch GD, SGD and Adam using validation RMSE in degrees Celsius. ',f'The lowest RMSE is {val(win["report"],"rmse_c"):.6f}°C for {win["effective_parameters"]["optimizer"]} with {win["penalty"]}.'],matrix)
 # Heatmaps show all error metrics, so every combination has visible MAE/RMSE/max.
 for key,label,unit in METRICS[:3]:
  matrix=np.array([[val(combo(kind,o,p)['report'],key) for p in PENS] for o in OPTS]);f,a=fig();im=a.imshow(matrix,cmap='YlGnBu',aspect='auto');a.set_xticks(range(4),PEN_LABELS);a.set_yticks(range(3),OPT_LABELS);a.set_xlabel('Regularization penalty');a.set_ylabel('Optimizer');f.colorbar(im,ax=a,label=label+' (°C)',shrink=.9)
  for row in range(3):
   for col in range(4):a.text(col,row,f'{matrix[row,col]:.4f}',ha='center',va='center',fontsize=11,color='white' if matrix[row,col]>(matrix.min()+matrix.max())/2 else '#162333')
  ii,jj=np.unravel_index(np.argmin(matrix),matrix.shape)
  add('Optimizer × regularization matrix',model+': full '+label+' heatmap',f,[f'Rows identify optimizers and columns identify penalties, with each cell showing validation {label} in degrees Celsius.',f'{OPT_LABELS[ii]} with {PEN_LABELS[jj]} has the lowest value at {matrix[ii,jj]:.6f}°C.'],dict(kind=kind,metric=key,values=matrix.tolist()))
# Paired architecture comparison with one clearly labeled row per setting.
f,a=fig();xs=[];ys=[];labs=[]
for oi,opt in enumerate(OPTS):
 for pi,pen in enumerate(PENS):
  x=val(combo('linear_gradient',opt,pen)['report'],'rmse_c');y=val(combo('mlp',opt,pen)['report'],'rmse_c');xs.append(x);ys.append(y);labs.append(OPT_LABELS[oi]+' / '+PEN_LABELS[pi])
for i,(x,y) in enumerate(zip(xs,ys)):a.plot([x,y],[i,i],color='#bbc5ce',linewidth=2)
a.scatter(xs,range(12),color=COLORS['linear_gradient'],label='Linear',s=30,zorder=3);a.scatter(ys,range(12),color=COLORS['mlp'],label='MLP',s=30,zorder=3)
a.set_yticks(range(12),labs);a.invert_yaxis();a.set_xlabel('Validation RMSE (°C)');a.legend(ncols=2,loc='upper right');a.set_xlim(1.4,7.6)
add('Optimizer × regularization matrix','Linear versus MLP under matched optimizer and penalty settings',f,['Each labeled row pairs linear and MLP results under one shared setting, with validation RMSE in degrees Celsius on the horizontal axis.','The best linear setting is Adam with L1 at 1.739552°C while the best MLP setting is Adam with L1+L2 at 2.672208°C.'],dict(labels=labs,linear_rmse=xs,mlp_rmse=ys))
# Direct pairwise native Lasso vs best custom L1 model.
lasso=next(x for x in U if x['run_name']=='lasso');dense=next(x for x in U if x['run_name']=='linear_gradient_adam_l1')
keys=[m[0] for m in METRICS];raw=np.array([[val(x['report'],k) for k in keys] for x in (lasso,dense)]);normalized=raw/raw.max(axis=0)
angles=np.linspace(0,2*np.pi,6,endpoint=False);ff=plt.figure(figsize=(7.4,3.25),layout='constrained');aa=ff.add_subplot(111,projection='polar')
for vals,label,color in zip(normalized,['Native Lasso','Linear Adam + L1'],['#159263','#875ca8']):
 aa.plot(np.r_[angles,angles[0]],np.r_[vals,vals[0]],label=label,color=color);aa.fill(np.r_[angles,angles[0]],np.r_[vals,vals[0]],alpha=.08,color=color)
aa.set_xticks(angles,['MAE','RMSE','Max error','Fit time','Latency','Size']);aa.set_ylim(0,1.05);aa.set_yticks([.25,.5,.75,1],['.25','.50','.75','1.00']);aa.legend(loc='upper right',bbox_to_anchor=(1.8,1.05))
add('Accuracy, latency and storage trade-offs','Native Lasso versus linear Adam + L1: normalized metrics',ff,['Each radial metric is divided by the worse value of these two configurations, so a smaller radius is better.','Lasso has RMSE 1.463817°C and an 11,197-byte bundle while linear Adam with L1 measures 1.739552°C and 21,566 bytes.'],dict(metric_order=keys,raw=raw.tolist(),normalized=normalized.tolist()))
for key,label,unit in [('median','Median latency','ms'),('size','Serialized size','bytes'),('train_seconds','Training time','s')]:
 f,a=fig();xx=[val(x['report'],key) for x in U];yy=[val(x['report'],'rmse_c') for x in U]
 for x,xv,yv in zip(U,xx,yy):a.scatter(xv,yv,c=COLORS[x['experiment']['kind']],s=45,alpha=.8,edgecolors='white',linewidth=.5)
 for x in (lasso,dense):
  xv=val(x['report'],key);yv=val(x['report'],'rmse_c');a.annotate(x['run_name'].replace('linear_gradient_','Linear '),(xv,yv),xytext=(7,-13 if x is lasso else 10),textcoords='offset points',fontsize=9,fontweight='bold')
 if key in ('median','size'):
  frontier=[]
  for i in sorted(range(len(xx)),key=lambda i:xx[i]):
   if not frontier or yy[i]<yy[frontier[-1]]:frontier.append(i)
  a.plot([xx[i] for i in frontier],[yy[i] for i in frontier],'k--',alpha=.5,label='Measured Pareto boundary');a.legend(loc='upper right')
 a.set_xscale('log');a.set_xlabel(label+' ('+unit+', log scale)');a.set_ylabel('Validation RMSE (°C)');a.grid(alpha=.15)
 add('Accuracy, latency and storage trade-offs','Validation accuracy versus '+label.lower(),f,[f'Each point is one of 32 unique fits with {label.lower()} on the logarithmic horizontal axis and RMSE on the vertical axis.',f'Lasso has the lowest RMSE at 1.463817°C with measured {label.lower()} of {fmt(val(lasso["report"],key))} {unit}.'],dict(runs=[x['run_name'] for x in U],x=xx,y=yy,x_unit=unit))
# Frozen-model validation vs final-test results.
f,a=fig();xx=np.arange(3)
for j,(part,color) in enumerate([('validation','#4275a6'),('test','#159263')]):
 vals=[final[part][k] for k,_,_ in METRICS[:3]];b=a.bar(xx+(j-.5)*.32,vals,width=.3,label=part.title(),color=color);a.bar_label(b,labels=[f'{v:.4f}' for v in vals],padding=3,fontsize=9)
a.set_xticks(xx,['MAE','RMSE','Maximum error']);a.set_ylabel('Temperature error (°C)');a.legend();a.set_ylim(0,6.5)
add('Frozen Lasso final evaluation','Aggregate validation versus held-out test errors',f,['Grouped bars compare validation and final-test MAE, RMSE and maximum error in degrees Celsius for the frozen Lasso.','Test RMSE is 1.339064°C versus validation 1.463817°C and test maximum error is 4.974435°C.'],dict(validation=[final['validation'][k] for k,_,_ in METRICS[:3]],test=[final['test'][k] for k,_,_ in METRICS[:3]]))
cores=list(final['test']['per_core'])
for key,label,unit in METRICS[:3]:
 f,a=fig();xx=np.arange(8)
 for j,(part,color) in enumerate([('validation','#4275a6'),('test','#159263')]):
  vals=[final[part]['per_core'][c][key] for c in cores];b=a.bar(xx+(j-.5)*.34,vals,width=.32,label=part.title(),color=color);a.bar_label(b,labels=[f'{v:.2f}' for v in vals],padding=2,fontsize=9)
 a.set_xticks(xx,[f'Core {i}' for i in range(8)]);a.set_ylabel(label+' (°C)');a.legend();a.set_ylim(0,max(final[part]['per_core'][c][key] for part in ('validation','test') for c in cores)*1.28)
 winner=min(cores,key=lambda c:final['test']['per_core'][c][key])
 add('Frozen Lasso final evaluation','Per-core validation versus test '+label,f,[f'Grouped bars compare {label} in degrees Celsius for each core between validation and final testing.',f'{winner} has the lowest test {label} at {final["test"]["per_core"][winner][key]:.6f}°C while Core_5_L0 has the largest test error.'],dict(cores=cores,metric=key,validation=[final['validation']['per_core'][c][key] for c in cores],test=[final['test']['per_core'][c][key] for c in cores]))
with (ROOT/'final_evaluation/test_predictions.csv').open() as stream:testrows=list(csv.DictReader(stream))
abs_error=np.array([float(x['absolute_error_c']) for x in testrows]).reshape(240,8);actual=np.array([float(x['actual_c']) for x in testrows]);predicted=np.array([float(x['predicted_c']) for x in testrows]);signed=predicted-actual
f,a=fig();a.boxplot([abs_error[:,i] for i in range(8)],tick_labels=[f'Core {i}' for i in range(8)],showfliers=True,patch_artist=True,boxprops=dict(facecolor='#cfe6dc'));a.set_ylabel('Absolute test error (°C)');a.set_xlabel('Core identifier')
add('Frozen Lasso final evaluation','Original per-sample test absolute-error distributions',f,['Boxes show medians and interquartile ranges with 1.5-IQR whiskers and outliers for 240 recorded absolute errors per core.','Core_5_L0 has the largest recorded absolute test error at 4.974435°C and no distribution samples were simulated.'],dict(source='final_evaluation/test_predictions.csv',counts=[240]*8,absolute_error_values=abs_error.T.tolist()))
f,a=fig();a.hist(signed,bins=30,color='#159263',edgecolor='white');a.set_xlabel('Predicted minus actual temperature (°C)');a.set_ylabel('Recorded core predictions (count)');a.axvline(0,color='#555',linestyle='--')
add('Frozen Lasso final evaluation','Signed errors from the 1,920 recorded test predictions',f,['The histogram counts actual recorded prediction errors on the vertical axis against signed temperature error on the horizontal axis.','The maximum absolute error is 4.974435°C across 1,920 comparisons and no fitted error distribution is assumed.'],dict(source='final_evaluation/test_predictions.csv',signed_error_values=signed.tolist(),bins=30))
f,a=fig();a.scatter(actual,predicted,s=7,alpha=.24,color='#159263',rasterized=True);low=min(actual.min(),predicted.min());high=max(actual.max(),predicted.max());a.plot([low,high],[low,high],'--',color='#555');a.set_xlabel('Supplied HotSpot label (°C)');a.set_ylabel('Frozen Lasso prediction (°C)')
add('Frozen Lasso final evaluation','Recorded test temperatures: predicted versus actual',f,['Each point is one recorded core prediction with the supplied HotSpot label horizontally and Lasso prediction vertically, while the dashed line marks equality.','The 1,920 points yield MAE 1.070281°C and RMSE 1.339064°C without retraining.'],dict(source='final_evaluation/test_predictions.csv',actual=actual.tolist(),predicted=predicted.tolist()))
# PDF layout: two large vector charts per page, captions always exactly two sentences.
for c in charts:c['captions']=[s.strip() for s in c['captions']];assert all(s.endswith('.') for s in c['captions'])
fontdir=Path('/usr/share/fonts/truetype/dejavu')
for name,file in [('DV','DejaVuSans.ttf'),('DVB','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyDV',fontName='DV',fontSize=10,leading=15,spaceAfter=10,textColor=HexColor('#263647')))
styles.add(ParagraphStyle(name='H1DV',fontName='DVB',fontSize=17,leading=23,spaceAfter=14,textColor=HexColor('#153e56')))
styles.add(ParagraphStyle(name='H2DV',fontName='DVB',fontSize=12,leading=16,spaceAfter=9,textColor=HexColor('#153e56')))
styles.add(ParagraphStyle(name='CapDV',fontName='DV',fontSize=9.5,leading=14,spaceAfter=12,textColor=HexColor('#344557')))
styles.add(ParagraphStyle(name='CellDV',fontName='DV',fontSize=9,leading=12))
styles.add(ParagraphStyle(name='TinyDV',fontName='DV',fontSize=9,leading=13,spaceAfter=8))
placements=[]
class Chart(Flowable):
 def __init__(self,index):super().__init__();self.index=index;self.width=480;self.height=211
 def draw(self):
  x,y=self.canv.absolutePosition(0,0);placements.append(dict(figure=self.index,page=self.canv.getPageNumber(),x=x,y=y,width=self.width,height=self.height))
class Doc(BaseDocTemplate):
 def beforeDocument(self):placements.clear()
 def afterFlowable(self,obj):
  if isinstance(obj,Paragraph) and obj.style.name=='H1DV' and obj.getPlainText() not in ('Contents','Stage B — Geometry-Aware ML Temperature Predictor','Complete Experimental Comparison'):
   name=f'heading-{self.page}-{hash(obj.getPlainText())}';self.canv.bookmarkPage(name);self.notify('TOCEntry',(0,obj.getPlainText(),self.page,name))
def footer(c,d):
 c.setStrokeColor(HexColor('#c6d3db'));c.line(49,42,A4[0]-49,42);c.setFont('DV',8);c.setFillColor(HexColor('#526575'));c.drawString(49,28,'Stage B • Frozen experiments • 18 November 2026');c.drawRightString(A4[0]-49,28,str(d.page))
base=WORK/'layout.pdf';doc=Doc(str(base),pagesize=A4,leftMargin=49,rightMargin=49,topMargin=46,bottomMargin=52,title='Stage B — Geometry-Aware ML Temperature Predictor: Complete Experimental Comparison',author='Lightweight Temperature Predictor project')
doc.addPageTemplates(PageTemplate(id='all',frames=[Frame(49,52,A4[0]-98,A4[1]-98,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
story=[]
def para(s,style='BodyDV'):return Paragraph(escape(s),styles[style])
def heading(s):story.append(para(s,'H1DV'))
def body(s):story.append(para(s))
story += [Spacer(1,95),para('Stage B — Geometry-Aware ML Temperature Predictor','H1DV'),para('Complete Experimental Comparison','H1DV'),Spacer(1,25),para('Seven model families · 40 additional comparison entries · 32 unique fits · frozen held-out evaluation','H2DV'),Spacer(1,30),para('Final model: Lasso regression, alpha = 0.001','H2DV'),para('Validation RMSE 1.463817°C · Test RMSE 1.339064°C','H2DV'),Spacer(1,35),para('Original measured records only. No training, tuning, new model selection or deployment was performed to prepare this report.'),PageBreak()]
heading('Contents');toc=TableOfContents();toc.levelStyles=[ParagraphStyle(name='toc',fontName='DV',fontSize=10,leading=16,leftIndent=0,firstLineIndent=0)];story.extend([toc,PageBreak()])
heading('Executive summary')
body('Lasso alpha=0.001 remains the approved model because it combines the lowest validation error with the smallest serialized bundle: 11,197 bytes. Its final test MAE is 1.070281°C, RMSE is 1.339064°C and maximum absolute error is 4.974435°C across 240 samples and 1,920 core-temperature comparisons.')
body('The original seven-model comparison and subsequent fixed optimizer/regularization experiments are reported separately. The latter contain 40 comparison entries: 6 optimizer, 10 regularization and 24 combinations, supported by 32 unique fits and 8 explicitly reused duplicate entries. Sixteen unique fits reached their 300-epoch budget; convergence is not assumed for those runs.')
body('Linear Adam with L1 is the strongest custom gradient-model alternative, with validation RMSE 1.739552°C and a 21,566-byte bundle. Its measured inference latency is lower than native Lasso, but microbenchmark timing is noisy and does not establish a deployment advantage.')
body('All previous experiment artifacts were recovered and checked against original checksum manifests. The final Lasso bundle is byte-identical to the tested bundle and reproduces all 1,920 saved final predictions exactly. No model was retrained and no other candidate was evaluated on the consumed test holdout.')
story.append(PageBreak());heading('Dataset, isolation and metrics')
body('Dataset: 77 heterogeneous floorplans b_fp_006 through b_fp_082, 20 power samples each, 1,540 CSV rows, 58 columns and 12,320 core-temperature labels. Every floorplan has 2 Big 4×4 mm and 6 Little 2×2 mm cores. Geometry, power and temperature units are mm, W and °C, with 45°C declared ambient. Original Stage A layouts and pilot layouts are excluded.')
body('The fixed seed-42 whole-floorplan split has 53 training, 12 validation and 12 test layouts, or 1,060/240/240 rows. The 252 geometry/power features and both input/target StandardScaler parameters are unchanged; scalers were fitted only on training rows. Selection used validation RMSE, and the frozen Lasso alone underwent final test evaluation.')
body('MAE is mean absolute error, RMSE is the square root of mean squared error and maximum error is the largest absolute error. Aggregate errors pool eight cores across all 240 validation or test rows; per-core errors use 240 values. Inference timing includes input scaling, prediction and inverse target scaling, but excludes CSV loading and feature engineering. Median/p95 timing uses 50 repetitions after warmup, and fit time measures estimator.fit only.')
body('Each plot uses a specific original measurement snapshot. The seven-family timings come from the original family comparison; the additional-entry timings come from the later optimizer/regularization phase. Small timing differences across phases are not interpreted as improvements. Serialized bundle size includes preprocessing metadata and is not runtime RAM.')
story.append(PageBreak())
current=None
for i,c in enumerate(charts):
 if i%2==0:
  if i:story.append(PageBreak())
  if c['section']!=current:heading(c['section']);current=c['section']
 block=[para(f'Figure {c["number"]}. {c["title"]}','H2DV'),Chart(c['number']),para(' '.join(c['captions']),'CapDV')]
 story.append(KeepTogether(block))
story.append(PageBreak());heading('Interpretation and limits')
body('The optimizer rates are the existing group defaults: Batch GD 0.01, SGD 0.001 and Adam 0.003, with a maximum 300 epochs. Batch GD uses all 1,060 training rows per update, SGD uses one row and Adam uses batches of 32. Patience is 20, tolerance is 1e-7 and gradient clipping is 10. Different update counts and fixed learning rates prevent claims of universal optimizer superiority.')
body('Custom linear/MLP combinations use L1 strength 0.0001, L2 strength 0.001 or both, with unpenalized biases. The objective is MSE averaged across rows and eight scaled targets plus l1·sum(abs(weights)) and l2·sum(weights²). L1 uses a sign subgradient. Combined L1+L2 is an elastic-net-style penalty, not the native sklearn ElasticNet estimator or an identical alpha convention.')
body('Native OLS uses least squares; native Ridge, Lasso and Elastic Net use their own supported solvers. They do not expose the framework’s Batch GD/SGD/Adam selector. Tree construction and boosting must not be labeled SGD/Adam. Random Forest depth/leaf settings are structural complexity controls, not L1/L2. XGBoost native penalties were not an additional sweep in the existing groups.')
body('The supplied labels are reported as original HotSpot results, but raw solver outputs and a complete matching simulation-provenance contract were unavailable for independent authentication. The combined CSV and original geometry/power hashes are preserved. The test sample contains only 12 unseen layouts; no universal accuracy guarantee or application threshold is claimed. The holdout is now consumed and must not guide further tuning.')
story.append(PageBreak());heading('Appendix A — Complete measured tables')
def table_rows(names,rows):
 cells=[[para(x,'CellDV') for x in names]]+[[para(str(x),'CellDV') for x in row] for row in rows]
 widths=[120]+[60]*6 if len(names)==7 else None
 t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#dce9ef')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,HexColor('#526575')),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f3f7f9')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));return t
def metric_table(entries):
 rows=[]
 for label,r in entries:rows.append([label,f'{val(r,"mae_c"):.6f}',f'{val(r,"rmse_c"):.6f}',f'{val(r,"max_absolute_error_c"):.6f}',f'{val(r,"train_seconds"):.6f}',f'{val(r,"median"):.6f}',f'{val(r,"size"):,}'])
 return table_rows(['Configuration','MAE °C','RMSE °C','Max °C','Fit s','Median ms','Bytes'],rows)
story.extend([para('A1. Original seven-model comparison','H2DV'),metric_table([(MODEL_NAMES[x['experiment']['name']],x) for x in C]),Spacer(1,16),para('All values are rounded for display; CSV/JSON preserve the original floating-point values. Latency p95 and batch measurements are preserved in the machine-readable records.','TinyDV'),PageBreak()])
for group in ('optimizers','regularization','combinations'):
 heading('Appendix A — '+group.replace('_',' ').title())
 entries=[x for x in R if x['group']==group]
 if group=='combinations':
  for ki,kind in enumerate(('linear_gradient','mlp')):
   if ki:story.append(PageBreak());heading('Appendix A — MLP combinations')
   es=[x for x in entries if x['experiment']['kind']==kind]
   labeled=[((('Linear' if kind=='linear_gradient' else 'MLP')+' / '+OPT_LABELS[OPTS.index(x['effective_parameters']['optimizer'])]+' / '+PEN_LABELS[PENS.index(x['penalty'])]+(' *' if x['reused'] else '')),x['report']) for x in es]
   story.append(metric_table(labeled));story.append(para('* Reused identical fit; canonical run and exact alias mapping are listed in combinations.csv.','TinyDV'))
 else:story.append(metric_table([(x['experiment']['name'].replace('linear_gradient_','Linear / ').replace('_',' '),x['report']) for x in entries]))
 story.append(PageBreak())
heading('Appendix B — Final Lasso metrics')
rows=[[str(i),f'{final["validation"]["per_core"][c]["rmse_c"]:.6f}',f'{final["test"]["per_core"][c]["mae_c"]:.6f}',f'{final["test"]["per_core"][c]["rmse_c"]:.6f}',f'{final["test"]["per_core"][c]["max_absolute_error_c"]:.6f}'] for i,c in enumerate(cores)]
story.append(table_rows(['Core','Val RMSE °C','Test MAE °C','Test RMSE °C','Test max °C'],rows));story.append(Spacer(1,15));body('Test latency: median 0.0632965 ms, p95 0.1411827 ms and 240-row batch 0.319088 ms. The frozen bundle remains exactly 11,197 bytes. Test aggregate RMSE is 8.52% lower than validation; individual Core 0, Core 4 and Core 5 RMSE values are higher than their validation values.');story.append(PageBreak())
heading('Appendix C — Complete exact configurations')
body('The entries below list effective estimator parameters, not newly tuned values. All experiments use seed 42 and the same saved preprocessing manifest. Native sklearn alpha values are not directly interchangeable with custom dense l1/l2 strengths.')
effective=read(ROOT/'seven_models/effective_parameters.json')
for name,record in effective.items():
 story.append(KeepTogether([para('Original family: '+MODEL_NAMES[name],'H2DV'),para(json.dumps(record,sort_keys=True),'TinyDV')]))
story.append(PageBreak());heading('Appendix C — Additional unique configurations')
for x in U:
 record=dict(parameters=x['effective_parameters'],training_details=x['training_details'])
 # Keep full per-core iteration arrays and exact all effective hyperparameters.
 story.append(KeepTogether([para(x['run_name'],'H2DV'),para(json.dumps(record,sort_keys=True),'TinyDV')]))
story.append(PageBreak());heading('Appendix D — Aliases, limits and provenance')
body('Reused entries reference the exact same newly measured fit, model bytes, timing and validation metrics; they are not independent repetitions. The following eight aliases preserve the original distinction between 40 entries and 32 fits.')
for x in R:
 if x['reused']:story.append(para(x['group']+' / '+x['experiment']['name']+' → '+x['run_name'],'TinyDV'))
body('The following 16 unique runs reached 300 epochs; those rows are budget-limited, not asserted converged.')
for x in U:
 if x['training_details'].get('epoch_cap_reached'):story.append(para(x['run_name'],'TinyDV'))
story.append(PageBreak());heading('Appendix E — Reproducibility and integrity')
body('Source: pardhu0716-dot/temperature-predictor, branch stage-b-ml-framework, experiment code/data commit 39b5494b5d9da5a6956aa572fb6b58c399a6f5ef. Runtime: Python 3.12.14, NumPy 2.3.5, sklearn 1.8.0, joblib 1.5.3 and CPU XGBoost 3.4.1; BLAS/OpenMP threads were each set to one. The presentation build does not fit estimators or invoke HotSpot.')
for label,value in [('Model SHA-256',final['model_sha256']),('Prepared manifest SHA-256',final['prepared_manifest_sha256']),('CSV SHA-256',core['results'][0]['source_csv_sha256'])]:body(label+': '+value)
body('The split manifest, all floorplan assignments, actual CSV schema, scaler parameters, 252-feature definitions and final model are preserved under models/final_lasso. Arrays are not duplicated in GitHub: recover their exact rows from the existing source CSV using the pinned preparation and seed. Original prediction CSVs preserve audit distributions; validation prediction records are losslessly gzip-compressed.')
body('The original 47 model/entry reports, frozen evaluation records, exact effective configurations, warnings and alias mappings are machine-readable under experiment_results. figure_data.json records every plotted value and caption. The repository preservation audit lists changed paths and confirms existing tracked blobs are unchanged. Only the approved final model bundle is committed; no intermediate estimators, caches or environments are included.')
body('The final test is consumed: subsequent development requires a new independent holdout or an explicitly revised protocol. No model should be selected based on the test plots in this report. Original measured records, rather than rounded labels, are authoritative for all numerical comparisons.')
# Preserve the full original latency columns, not only the median shown in main tables.
story.append(PageBreak());heading('Appendix F — Complete latency records')
def latency_table(entries):
 rows=[]
 for label,r in entries:
  l=r['latency'];rows.append([label,f"{l['single_sample_median_ms']:.6f}",f"{l['single_sample_p95_ms']:.6f}",f"{l['batch_ms']:.6f}",str(l['repeats'])])
 cells=[[para(x,'CellDV') for x in ['Configuration','Median ms','p95 ms','Batch 240 ms','Repeats']]]+[[para(x,'CellDV') for x in row] for row in rows]
 t=Table(cells,colWidths=[160,80,80,90,70],repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#dce9ef')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f3f7f9')]),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));return t
story.append(para('Original seven-model timing snapshot','H2DV'));story.append(latency_table([(MODEL_NAMES[x['experiment']['name']],x) for x in C]))
for group in ('optimizers','regularization','combinations'):
 entries=[x for x in R if x['group']==group]
 parts=[entries] if group!='combinations' else [[x for x in entries if x['experiment']['kind']==kind] for kind in ('linear_gradient','mlp')]
 for part in parts:
  story.append(PageBreak());heading('Appendix F — '+group.title()+' latency')
  labels=[]
  for x in part:
   if group=='combinations':label=MODEL_NAMES[x['experiment']['kind']]+' / '+OPT_LABELS[OPTS.index(x['effective_parameters']['optimizer'])]+' / '+PEN_LABELS[PENS.index(x['penalty'])]
   else:label=x['experiment']['name'].replace('linear_gradient_','Linear / ').replace('_',' ')
   labels.append((label+(' *' if x['reused'] else ''),x['report']))
  story.append(latency_table(labels));story.append(para('* Identical fit reused; p95 and batch timing are the original canonical measurements, not independent repetitions.','TinyDV'))

doc.multiBuild(story)
# Overlay pure-vector chart PDFs into their reserved rectangles, without reflow.
pdf=fitz.open(base)
for place in placements:
 page=pdf[place['page']-1];chart=charts[place['figure']-1];source=fitz.open(chart['figure_file'])
 rect=fitz.Rect(place['x'],A4[1]-place['y']-place['height'],place['x']+place['width'],A4[1]-place['y'])
 page.show_pdf_page(rect,source,0,keep_proportion=True);source.close()
output=ROOT/'Stage_B_Complete_Experimental_Comparison.pdf';pdf.save(output,garbage=4,deflate=True);pdf.close()
records=[{k:v for k,v in c.items() if k!='figure_file'} for c in charts]
(ROOT/'figure_data.json').write_text(json.dumps(records,indent=2))
(WORK/'placements.json').write_text(json.dumps(placements,indent=2))
print(json.dumps(dict(pdf=str(output),figures=len(charts),pages=len(fitz.open(output)),bytes=output.stat().st_size,placements=len(placements))))
