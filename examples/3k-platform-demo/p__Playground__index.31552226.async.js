"use strict";(self.webpackChunkant_design_pro=self.webpackChunkant_design_pro||[]).push([[3709],{58509:function(V,u,n){n.r(u),n.d(u,{default:function(){return W}});var N=n(15009),m=n.n(N),T=n(99289),Z=n.n(T),B=n(5574),c=n.n(B),E=n(90930),h=n(2453),L=n(85696),O=n(75398),i=n(67294),A=n(83065),H=n(21373),F=n(86032),a={container:"container___YSjHW",topSection:"topSection___LuO_8",mainContent:"mainContent___YbW2M",leftSection:"leftSection___KsJk0",chatContainer:"chatContainer___yleFo",rightSection:"rightSection___oifvM",copyButton:"copyButton___cnLGo"},G=n(85175),I=n(76772),e=n(85893),R=function(){var d=(0,I.useIntl)(),$=(0,i.useState)([]),f=c()($,2),v=f[0],z=f[1],U=(0,i.useState)(),p=c()(U,2),g=p[0],y=p[1],Y=(0,i.useState)(),S=c()(Y,2),C=S[0],M=S[1],w=(0,i.useState)(!1),j=c()(w,2),D=j[0],x=j[1],J=function(){var o=Z()(m()().mark(function r(){var s,l;return m()().wrap(function(t){for(;;)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,(0,A.$7)();case 3:l=t.sent,z(l.data||[]),((s=l.data)===null||s===void 0?void 0:s.length)>0&&(y(l.data[0].model_name),M("/chat-trial?model=".concat(l.data[0].model_name))),t.next=11;break;case 8:t.prev=8,t.t0=t.catch(0),console.error("Failed to fetch models:",t.t0);case 11:case"end":return t.stop()}},r,null,[[0,8]])}));return function(){return o.apply(this,arguments)}}();(0,i.useEffect)(function(){J()},[]);var K=function(r){y(r);var s=v.find(function(l){return l.model_name===r});M("/chat-trial?model=".concat(s==null?void 0:s.model_name))},P=`from openai import OpenAI

client = OpenAI(
    base_url="`.concat(window.location.origin,`/api/v1",
    api_key="dummy", # \u7B97\u60F3\u4E91token
)

response = client.chat.completions.create(
    model="`).concat(g,`",
    messages=[
        {"role": "user", "content": "How to learn python?"}
    ],
    max_tokens=200,
    temperature=0.7,
    top_p=1,
    stream=True,
)

for chunk in response:
    if not chunk.choices:
        continue
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
`),Q=function(){navigator.clipboard.writeText(P).then(function(){h.ZP.success(d.formatMessage({id:"playground.copy.success"}))}).catch(function(){h.ZP.error(d.formatMessage({id:"playground.copy.failed"}))})};return(0,e.jsx)(E._z,{children:(0,e.jsxs)("div",{className:a.container,children:[(0,e.jsx)("div",{className:a.topSection,children:(0,e.jsx)(L.Z,{style:{width:200},value:g,onChange:K,options:v.map(function(o){return{label:o.model_name,value:o.model_name}})})}),(0,e.jsxs)("div",{className:a.mainContent,children:[(0,e.jsx)("div",{className:a.leftSection,children:(0,e.jsx)("div",{className:a.chatContainer,children:C&&(0,e.jsx)("iframe",{src:C,style:{width:"100%",height:"100%",border:"none",transform:"scale(1)",transformOrigin:"0 0",minWidth:"100%",minHeight:"100%"},sandbox:"allow-same-origin allow-scripts allow-popups allow-forms allow-modals",allow:"camera *; microphone *",referrerPolicy:"origin",onLoad:function(){return console.log("iframe loaded")},onError:function(r){return console.error("iframe error:",r)}})})}),(0,e.jsxs)("div",{className:a.rightSection,onMouseEnter:function(){return x(!0)},onMouseLeave:function(){return x(!1)},children:[D&&(0,e.jsx)(O.ZP,{className:a.copyButton,icon:(0,e.jsx)(G.Z,{}),size:"small",onClick:Q,children:d.formatMessage({id:"playground.copy.button"})}),(0,e.jsx)(H.Z,{language:"python",style:F.Z,customStyle:{margin:0,borderRadius:"8px"},codeTagProps:{style:{lineHeight:"1"}},children:P})]})]})]})})},W=R}}]);
