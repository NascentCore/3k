(self.webpackChunkant_design_pro=self.webpackChunkant_design_pro||[]).push([[8882],{59542:function(Ge){(function(ie,V){Ge.exports=V()})(this,function(){"use strict";var ie="day";return function(V,We,M){var n=function(B){return B.add(4-B.isoWeekday(),ie)},Q=We.prototype;Q.isoWeekYear=function(){return n(this).year()},Q.isoWeek=function(B){if(!this.$utils().u(B))return this.add(7*(B-this.isoWeek()),ie);var q,$,J,b,ce=n(this),De=(q=this.isoWeekYear(),$=this.$u,J=($?M.utc:M)().year(q).startOf("year"),b=4-J.isoWeekday(),J.isoWeekday()>4&&(b+=7),J.add(b,ie));return ce.diff(De,"week")+1},Q.isoWeekday=function(B){return this.$utils().u(B)?this.day()||7:this.day(this.day()%7?B:B-7)};var O=Q.startOf;Q.startOf=function(B,q){var $=this.$utils(),J=!!$.u(q)||q;return $.p(B)==="isoweek"?J?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):O.bind(this)(B,q)}}})},88882:function(Ge,ie,V){"use strict";V.d(ie,{diagram:function(){return Zt}});var We=V(64348),M=V(22957),n=V(35096),Q=V(17967),O=V(27484),B=V(59542),q=V(10285),$=V(28734),J=V(1646),b=V(989),ce=function(){var e=(0,n.eW)(function(h,o,l,d){for(l=l||{},d=h.length;d--;l[h[d]]=o);return l},"o"),s=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],r=[1,26],c=[1,27],a=[1,28],f=[1,29],g=[1,30],C=[1,31],F=[1,32],G=[1,33],E=[1,34],P=[1,9],X=[1,10],N=[1,11],U=[1,12],w=[1,13],de=[1,14],fe=[1,15],ke=[1,16],he=[1,19],ae=[1,20],me=[1,21],ye=[1,22],ge=[1,23],ve=[1,25],y=[1,35],p={trace:(0,n.eW)(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:(0,n.eW)(function(o,l,d,u,m,i,W){var t=i.length-1;switch(m){case 1:return i[t-1];case 2:this.$=[];break;case 3:i[t-1].push(i[t]),this.$=i[t-1];break;case 4:case 5:this.$=i[t];break;case 6:case 7:this.$=[];break;case 8:u.setWeekday("monday");break;case 9:u.setWeekday("tuesday");break;case 10:u.setWeekday("wednesday");break;case 11:u.setWeekday("thursday");break;case 12:u.setWeekday("friday");break;case 13:u.setWeekday("saturday");break;case 14:u.setWeekday("sunday");break;case 15:u.setWeekend("friday");break;case 16:u.setWeekend("saturday");break;case 17:u.setDateFormat(i[t].substr(11)),this.$=i[t].substr(11);break;case 18:u.enableInclusiveEndDates(),this.$=i[t].substr(18);break;case 19:u.TopAxis(),this.$=i[t].substr(8);break;case 20:u.setAxisFormat(i[t].substr(11)),this.$=i[t].substr(11);break;case 21:u.setTickInterval(i[t].substr(13)),this.$=i[t].substr(13);break;case 22:u.setExcludes(i[t].substr(9)),this.$=i[t].substr(9);break;case 23:u.setIncludes(i[t].substr(9)),this.$=i[t].substr(9);break;case 24:u.setTodayMarker(i[t].substr(12)),this.$=i[t].substr(12);break;case 27:u.setDiagramTitle(i[t].substr(6)),this.$=i[t].substr(6);break;case 28:this.$=i[t].trim(),u.setAccTitle(this.$);break;case 29:case 30:this.$=i[t].trim(),u.setAccDescription(this.$);break;case 31:u.addSection(i[t].substr(8)),this.$=i[t].substr(8);break;case 33:u.addTask(i[t-1],i[t]),this.$="task";break;case 34:this.$=i[t-1],u.setClickEvent(i[t-1],i[t],null);break;case 35:this.$=i[t-2],u.setClickEvent(i[t-2],i[t-1],i[t]);break;case 36:this.$=i[t-2],u.setClickEvent(i[t-2],i[t-1],null),u.setLink(i[t-2],i[t]);break;case 37:this.$=i[t-3],u.setClickEvent(i[t-3],i[t-2],i[t-1]),u.setLink(i[t-3],i[t]);break;case 38:this.$=i[t-2],u.setClickEvent(i[t-2],i[t],null),u.setLink(i[t-2],i[t-1]);break;case 39:this.$=i[t-3],u.setClickEvent(i[t-3],i[t-1],i[t]),u.setLink(i[t-3],i[t-2]);break;case 40:this.$=i[t-1],u.setLink(i[t-1],i[t]);break;case 41:case 47:this.$=i[t-1]+" "+i[t];break;case 42:case 43:case 45:this.$=i[t-2]+" "+i[t-1]+" "+i[t];break;case 44:case 46:this.$=i[t-3]+" "+i[t-2]+" "+i[t-1]+" "+i[t];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},e(s,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:r,13:c,14:a,15:f,16:g,17:C,18:F,19:18,20:G,21:E,22:P,23:X,24:N,25:U,26:w,27:de,28:fe,29:ke,30:he,31:ae,33:me,35:ye,36:ge,37:24,38:ve,40:y},e(s,[2,7],{1:[2,1]}),e(s,[2,3]),{9:36,11:17,12:r,13:c,14:a,15:f,16:g,17:C,18:F,19:18,20:G,21:E,22:P,23:X,24:N,25:U,26:w,27:de,28:fe,29:ke,30:he,31:ae,33:me,35:ye,36:ge,37:24,38:ve,40:y},e(s,[2,5]),e(s,[2,6]),e(s,[2,17]),e(s,[2,18]),e(s,[2,19]),e(s,[2,20]),e(s,[2,21]),e(s,[2,22]),e(s,[2,23]),e(s,[2,24]),e(s,[2,25]),e(s,[2,26]),e(s,[2,27]),{32:[1,37]},{34:[1,38]},e(s,[2,30]),e(s,[2,31]),e(s,[2,32]),{39:[1,39]},e(s,[2,8]),e(s,[2,9]),e(s,[2,10]),e(s,[2,11]),e(s,[2,12]),e(s,[2,13]),e(s,[2,14]),e(s,[2,15]),e(s,[2,16]),{41:[1,40],43:[1,41]},e(s,[2,4]),e(s,[2,28]),e(s,[2,29]),e(s,[2,33]),e(s,[2,34],{42:[1,42],43:[1,43]}),e(s,[2,40],{41:[1,44]}),e(s,[2,35],{43:[1,45]}),e(s,[2,36]),e(s,[2,38],{42:[1,46]}),e(s,[2,37]),e(s,[2,39])],defaultActions:{},parseError:(0,n.eW)(function(o,l){if(l.recoverable)this.trace(o);else{var d=new Error(o);throw d.hash=l,d}},"parseError"),parse:(0,n.eW)(function(o){var l=this,d=[0],u=[],m=[null],i=[],W=this.table,t="",k=0,D=0,x=0,_=2,I=1,S=i.slice.call(arguments,1),A=Object.create(this.lexer),Z={yy:{}};for(var Ne in this.yy)Object.prototype.hasOwnProperty.call(this.yy,Ne)&&(Z.yy[Ne]=this.yy[Ne]);A.setInput(o,Z.yy),Z.yy.lexer=A,Z.yy.parser=this,typeof A.yylloc=="undefined"&&(A.yylloc={});var Ue=A.yylloc;i.push(Ue);var Ht=A.options&&A.options.ranges;typeof Z.yy.parseError=="function"?this.parseError=Z.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function qt(Y){d.length=d.length-2*Y,m.length=m.length-Y,i.length=i.length-Y}(0,n.eW)(qt,"popStack");function rt(){var Y;return Y=u.pop()||A.lex()||I,typeof Y!="number"&&(Y instanceof Array&&(u=Y,Y=u.pop()),Y=l.symbols_[Y]||Y),Y}(0,n.eW)(rt,"lex");for(var R,je,te,j,Jt,ze,ne={},_e,H,at,we;;){if(te=d[d.length-1],this.defaultActions[te]?j=this.defaultActions[te]:((R===null||typeof R=="undefined")&&(R=rt()),j=W[te]&&W[te][R]),typeof j=="undefined"||!j.length||!j[0]){var Ke="";we=[];for(_e in W[te])this.terminals_[_e]&&_e>_&&we.push("'"+this.terminals_[_e]+"'");A.showPosition?Ke="Parse error on line "+(k+1)+`:
`+A.showPosition()+`
Expecting `+we.join(", ")+", got '"+(this.terminals_[R]||R)+"'":Ke="Parse error on line "+(k+1)+": Unexpected "+(R==I?"end of input":"'"+(this.terminals_[R]||R)+"'"),this.parseError(Ke,{text:A.match,token:this.terminals_[R]||R,line:A.yylineno,loc:Ue,expected:we})}if(j[0]instanceof Array&&j.length>1)throw new Error("Parse Error: multiple actions possible at state: "+te+", token: "+R);switch(j[0]){case 1:d.push(R),m.push(A.yytext),i.push(A.yylloc),d.push(j[1]),R=null,je?(R=je,je=null):(D=A.yyleng,t=A.yytext,k=A.yylineno,Ue=A.yylloc,x>0&&x--);break;case 2:if(H=this.productions_[j[1]][1],ne.$=m[m.length-H],ne._$={first_line:i[i.length-(H||1)].first_line,last_line:i[i.length-1].last_line,first_column:i[i.length-(H||1)].first_column,last_column:i[i.length-1].last_column},Ht&&(ne._$.range=[i[i.length-(H||1)].range[0],i[i.length-1].range[1]]),ze=this.performAction.apply(ne,[t,D,k,Z.yy,j[1],m,i].concat(S)),typeof ze!="undefined")return ze;H&&(d=d.slice(0,-1*H*2),m=m.slice(0,-1*H),i=i.slice(0,-1*H)),d.push(this.productions_[j[1]][0]),m.push(ne.$),i.push(ne._$),at=W[d[d.length-2]][d[d.length-1]],d.push(at);break;case 3:return!0}}return!0},"parse")},T=function(){var h={EOF:1,parseError:(0,n.eW)(function(l,d){if(this.yy.parser)this.yy.parser.parseError(l,d);else throw new Error(l)},"parseError"),setInput:(0,n.eW)(function(o,l){return this.yy=l||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:(0,n.eW)(function(){var o=this._input[0];this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o;var l=o.match(/(?:\r\n?|\n).*/g);return l?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:(0,n.eW)(function(o){var l=o.length,d=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-l),this.offset-=l;var u=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),d.length-1&&(this.yylineno-=d.length-1);var m=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:d?(d.length===u.length?this.yylloc.first_column:0)+u[u.length-d.length].length-d[0].length:this.yylloc.first_column-l},this.options.ranges&&(this.yylloc.range=[m[0],m[0]+this.yyleng-l]),this.yyleng=this.yytext.length,this},"unput"),more:(0,n.eW)(function(){return this._more=!0,this},"more"),reject:(0,n.eW)(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:(0,n.eW)(function(o){this.unput(this.match.slice(o))},"less"),pastInput:(0,n.eW)(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:(0,n.eW)(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:(0,n.eW)(function(){var o=this.pastInput(),l=new Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+l+"^"},"showPosition"),test_match:(0,n.eW)(function(o,l){var d,u,m;if(this.options.backtrack_lexer&&(m={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(m.yylloc.range=this.yylloc.range.slice(0))),u=o[0].match(/(?:\r\n?|\n).*/g),u&&(this.yylineno+=u.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:u?u[u.length-1].length-u[u.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],d=this.performAction.call(this,this.yy,this,l,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),d)return d;if(this._backtrack){for(var i in m)this[i]=m[i];return!1}return!1},"test_match"),next:(0,n.eW)(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var o,l,d,u;this._more||(this.yytext="",this.match="");for(var m=this._currentRules(),i=0;i<m.length;i++)if(d=this._input.match(this.rules[m[i]]),d&&(!l||d[0].length>l[0].length)){if(l=d,u=i,this.options.backtrack_lexer){if(o=this.test_match(d,m[i]),o!==!1)return o;if(this._backtrack){l=!1;continue}else return!1}else if(!this.options.flex)break}return l?(o=this.test_match(l,m[u]),o!==!1?o:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:(0,n.eW)(function(){var l=this.next();return l||this.lex()},"lex"),begin:(0,n.eW)(function(l){this.conditionStack.push(l)},"begin"),popState:(0,n.eW)(function(){var l=this.conditionStack.length-1;return l>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:(0,n.eW)(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:(0,n.eW)(function(l){return l=this.conditionStack.length-1-Math.abs(l||0),l>=0?this.conditionStack[l]:"INITIAL"},"topState"),pushState:(0,n.eW)(function(l){this.begin(l)},"pushState"),stateStackSize:(0,n.eW)(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:(0,n.eW)(function(l,d,u,m){var i=m;switch(u){case 0:return this.begin("open_directive"),"open_directive";break;case 1:return this.begin("acc_title"),31;break;case 2:return this.popState(),"acc_title_value";break;case 3:return this.begin("acc_descr"),33;break;case 4:return this.popState(),"acc_descr_value";break;case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return h}();p.lexer=T;function v(){this.yy={}}return(0,n.eW)(v,"Parser"),v.prototype=p,p.Parser=v,new v}();ce.parser=ce;var De=ce;O.extend(B),O.extend(q),O.extend($);var Xe={friday:5,saturday:6},z="",Ee="",Ce=void 0,Se="",oe=[],le=[],Ie=new Map,Ae=[],pe=[],se="",Oe="",Ze=["active","done","crit","milestone","vert"],Le=[],ue=!1,Me=!1,Fe="sunday",Te="saturday",Pe=0,nt=(0,n.eW)(function(){Ae=[],pe=[],se="",Le=[],be=0,Ve=void 0,xe=void 0,L=[],z="",Ee="",Oe="",Ce=void 0,Se="",oe=[],le=[],ue=!1,Me=!1,Pe=0,Ie=new Map,(0,M.ZH)(),Fe="sunday",Te="saturday"},"clear"),ct=(0,n.eW)(function(e){Ee=e},"setAxisFormat"),ot=(0,n.eW)(function(){return Ee},"getAxisFormat"),lt=(0,n.eW)(function(e){Ce=e},"setTickInterval"),ut=(0,n.eW)(function(){return Ce},"getTickInterval"),dt=(0,n.eW)(function(e){Se=e},"setTodayMarker"),ft=(0,n.eW)(function(){return Se},"getTodayMarker"),kt=(0,n.eW)(function(e){z=e},"setDateFormat"),ht=(0,n.eW)(function(){ue=!0},"enableInclusiveEndDates"),mt=(0,n.eW)(function(){return ue},"endDatesAreInclusive"),yt=(0,n.eW)(function(){Me=!0},"enableTopAxis"),gt=(0,n.eW)(function(){return Me},"topAxisEnabled"),vt=(0,n.eW)(function(e){Oe=e},"setDisplayMode"),pt=(0,n.eW)(function(){return Oe},"getDisplayMode"),Tt=(0,n.eW)(function(){return z},"getDateFormat"),bt=(0,n.eW)(function(e){oe=e.toLowerCase().split(/[\s,]+/)},"setIncludes"),xt=(0,n.eW)(function(){return oe},"getIncludes"),_t=(0,n.eW)(function(e){le=e.toLowerCase().split(/[\s,]+/)},"setExcludes"),wt=(0,n.eW)(function(){return le},"getExcludes"),Wt=(0,n.eW)(function(){return Ie},"getLinks"),Dt=(0,n.eW)(function(e){se=e,Ae.push(e)},"addSection"),Et=(0,n.eW)(function(){return Ae},"getSections"),Ct=(0,n.eW)(function(){let e=et();const s=10;let r=0;for(;!e&&r<s;)e=et(),r++;return pe=L,pe},"getTasks"),He=(0,n.eW)(function(e,s,r,c){const a=e.format(s.trim()),f=e.format("YYYY-MM-DD");return c.includes(a)||c.includes(f)?!1:r.includes("weekends")&&(e.isoWeekday()===Xe[Te]||e.isoWeekday()===Xe[Te]+1)||r.includes(e.format("dddd").toLowerCase())?!0:r.includes(a)||r.includes(f)},"isInvalidDate"),St=(0,n.eW)(function(e){Fe=e},"setWeekday"),It=(0,n.eW)(function(){return Fe},"getWeekday"),At=(0,n.eW)(function(e){Te=e},"setWeekend"),qe=(0,n.eW)(function(e,s,r,c){if(!r.length||e.manualEndTime)return;let a;e.startTime instanceof Date?a=O(e.startTime):a=O(e.startTime,s,!0),a=a.add(1,"d");let f;e.endTime instanceof Date?f=O(e.endTime):f=O(e.endTime,s,!0);const[g,C]=Ot(a,f,s,r,c);e.endTime=g.toDate(),e.renderEndTime=C},"checkTaskDates"),Ot=(0,n.eW)(function(e,s,r,c,a){let f=!1,g=null;for(;e<=s;)f||(g=s.toDate()),f=He(e,r,c,a),f&&(s=s.add(1,"d")),e=e.add(1,"d");return[s,g]},"fixTaskDates"),Re=(0,n.eW)(function(e,s,r){if(r=r.trim(),(0,n.eW)(C=>{const F=C.trim();return F==="x"||F==="X"},"isTimestampFormat")(s)&&/^\d+$/.test(r))return new Date(Number(r));const f=new RegExp("^after\\s+(?<ids>[\\d\\w- ]+)").exec(r);if(f!==null){let C=null;for(const G of f.groups.ids.split(" ")){let E=ee(G);E!==void 0&&(!C||E.endTime>C.endTime)&&(C=E)}if(C)return C.endTime;const F=new Date;return F.setHours(0,0,0,0),F}let g=O(r,s.trim(),!0);if(g.isValid())return g.toDate();{n.cM.debug("Invalid date:"+r),n.cM.debug("With date format:"+s.trim());const C=new Date(r);if(C===void 0||isNaN(C.getTime())||C.getFullYear()<-1e4||C.getFullYear()>1e4)throw new Error("Invalid date:"+r);return C}},"getStartDate"),Je=(0,n.eW)(function(e){const s=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(e.trim());return s!==null?[Number.parseFloat(s[1]),s[2]]:[NaN,"ms"]},"parseDuration"),Qe=(0,n.eW)(function(e,s,r,c=!1){r=r.trim();const f=new RegExp("^until\\s+(?<ids>[\\d\\w- ]+)").exec(r);if(f!==null){let E=null;for(const X of f.groups.ids.split(" ")){let N=ee(X);N!==void 0&&(!E||N.startTime<E.startTime)&&(E=N)}if(E)return E.startTime;const P=new Date;return P.setHours(0,0,0,0),P}let g=O(r,s.trim(),!0);if(g.isValid())return c&&(g=g.add(1,"d")),g.toDate();let C=O(e);const[F,G]=Je(r);if(!Number.isNaN(F)){const E=C.add(F,G);E.isValid()&&(C=E)}return C.toDate()},"getEndDate"),be=0,re=(0,n.eW)(function(e){return e===void 0?(be=be+1,"task"+be):e},"parseId"),Lt=(0,n.eW)(function(e,s){let r;s.substr(0,1)===":"?r=s.substr(1,s.length):r=s;const c=r.split(","),a={};Be(c,a,Ze);for(let g=0;g<c.length;g++)c[g]=c[g].trim();let f="";switch(c.length){case 1:a.id=re(),a.startTime=e.endTime,f=c[0];break;case 2:a.id=re(),a.startTime=Re(void 0,z,c[0]),f=c[1];break;case 3:a.id=re(c[0]),a.startTime=Re(void 0,z,c[1]),f=c[2];break;default:}return f&&(a.endTime=Qe(a.startTime,z,f,ue),a.manualEndTime=O(f,"YYYY-MM-DD",!0).isValid(),qe(a,z,le,oe)),a},"compileData"),Mt=(0,n.eW)(function(e,s){let r;s.substr(0,1)===":"?r=s.substr(1,s.length):r=s;const c=r.split(","),a={};Be(c,a,Ze);for(let f=0;f<c.length;f++)c[f]=c[f].trim();switch(c.length){case 1:a.id=re(),a.startTime={type:"prevTaskEnd",id:e},a.endTime={data:c[0]};break;case 2:a.id=re(),a.startTime={type:"getStartDate",startData:c[0]},a.endTime={data:c[1]};break;case 3:a.id=re(c[0]),a.startTime={type:"getStartDate",startData:c[1]},a.endTime={data:c[2]};break;default:}return a},"parseData"),Ve,xe,L=[],$e={},Ft=(0,n.eW)(function(e,s){const r={section:se,type:se,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:s},task:e,classes:[]},c=Mt(xe,s);r.raw.startTime=c.startTime,r.raw.endTime=c.endTime,r.id=c.id,r.prevTaskId=xe,r.active=c.active,r.done=c.done,r.crit=c.crit,r.milestone=c.milestone,r.vert=c.vert,r.order=Pe,Pe++;const a=L.push(r);xe=r.id,$e[r.id]=a-1},"addTask"),ee=(0,n.eW)(function(e){const s=$e[e];return L[s]},"findTaskById"),Pt=(0,n.eW)(function(e,s){const r={section:se,type:se,description:e,task:e,classes:[]},c=Lt(Ve,s);r.startTime=c.startTime,r.endTime=c.endTime,r.id=c.id,r.active=c.active,r.done=c.done,r.crit=c.crit,r.milestone=c.milestone,r.vert=c.vert,Ve=r,pe.push(r)},"addTaskOrg"),et=(0,n.eW)(function(){const e=(0,n.eW)(function(r){const c=L[r];let a="";switch(L[r].raw.startTime.type){case"prevTaskEnd":{const f=ee(c.prevTaskId);c.startTime=f.endTime;break}case"getStartDate":a=Re(void 0,z,L[r].raw.startTime.startData),a&&(L[r].startTime=a);break}return L[r].startTime&&(L[r].endTime=Qe(L[r].startTime,z,L[r].raw.endTime.data,ue),L[r].endTime&&(L[r].processed=!0,L[r].manualEndTime=O(L[r].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),qe(L[r],z,le,oe))),L[r].processed},"compileTask");let s=!0;for(const[r,c]of L.entries())e(r),s=s&&c.processed;return s},"compileTasks"),Rt=(0,n.eW)(function(e,s){let r=s;(0,M.nV)().securityLevel!=="loose"&&(r=(0,Q.N)(s)),e.split(",").forEach(function(c){ee(c)!==void 0&&(it(c,()=>{window.open(r,"_self")}),Ie.set(c,r))}),tt(e,"clickable")},"setLink"),tt=(0,n.eW)(function(e,s){e.split(",").forEach(function(r){let c=ee(r);c!==void 0&&c.classes.push(s)})},"setClass"),Vt=(0,n.eW)(function(e,s,r){if((0,M.nV)().securityLevel!=="loose"||s===void 0)return;let c=[];if(typeof r=="string"){c=r.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let f=0;f<c.length;f++){let g=c[f].trim();g.startsWith('"')&&g.endsWith('"')&&(g=g.substr(1,g.length-2)),c[f]=g}}c.length===0&&c.push(e),ee(e)!==void 0&&it(e,()=>{We.w8.runFunc(s,...c)})},"setClickFun"),it=(0,n.eW)(function(e,s){Le.push(function(){const r=document.querySelector(`[id="${e}"]`);r!==null&&r.addEventListener("click",function(){s()})},function(){const r=document.querySelector(`[id="${e}-text"]`);r!==null&&r.addEventListener("click",function(){s()})})},"pushFun"),Bt=(0,n.eW)(function(e,s,r){e.split(",").forEach(function(c){Vt(c,s,r)}),tt(e,"clickable")},"setClickEvent"),Yt=(0,n.eW)(function(e){Le.forEach(function(s){s(e)})},"bindFunctions"),Nt={getConfig:(0,n.eW)(()=>(0,M.nV)().gantt,"getConfig"),clear:nt,setDateFormat:kt,getDateFormat:Tt,enableInclusiveEndDates:ht,endDatesAreInclusive:mt,enableTopAxis:yt,topAxisEnabled:gt,setAxisFormat:ct,getAxisFormat:ot,setTickInterval:lt,getTickInterval:ut,setTodayMarker:dt,getTodayMarker:ft,setAccTitle:M.GN,getAccTitle:M.eu,setDiagramTitle:M.g2,getDiagramTitle:M.Kr,setDisplayMode:vt,getDisplayMode:pt,setAccDescription:M.U$,getAccDescription:M.Mx,addSection:Dt,getSections:Et,getTasks:Ct,addTask:Ft,findTaskById:ee,addTaskOrg:Pt,setIncludes:bt,getIncludes:xt,setExcludes:_t,getExcludes:wt,setClickEvent:Bt,setLink:Rt,getLinks:Wt,bindFunctions:Yt,parseDuration:Je,isInvalidDate:He,setWeekday:St,getWeekday:It,setWeekend:At};function Be(e,s,r){let c=!0;for(;c;)c=!1,r.forEach(function(a){const f="^\\s*"+a+"\\s*$",g=new RegExp(f);e[0].match(g)&&(s[a]=!0,e.shift(1),c=!0)})}(0,n.eW)(Be,"getTaskTags"),O.extend(J);var Ut=(0,n.eW)(function(){n.cM.debug("Something is calling, setConf, remove the call")},"setConf"),st={monday:b.Ox9,tuesday:b.YDX,wednesday:b.EFj,thursday:b.Igq,friday:b.y2j,saturday:b.LqH,sunday:b.Zyz},jt=(0,n.eW)((e,s)=>{let r=[...e].map(()=>-1/0),c=[...e].sort((f,g)=>f.startTime-g.startTime||f.order-g.order),a=0;for(const f of c)for(let g=0;g<r.length;g++)if(f.startTime>=r[g]){r[g]=f.endTime,f.order=g+s,g>a&&(a=g);break}return a},"getMaxIntersections"),K,Ye=1e4,zt=(0,n.eW)(function(e,s,r,c){const a=(0,M.nV)().gantt,f=(0,M.nV)().securityLevel;let g;f==="sandbox"&&(g=(0,b.Ys)("#i"+s));const C=f==="sandbox"?(0,b.Ys)(g.nodes()[0].contentDocument.body):(0,b.Ys)("body"),F=f==="sandbox"?g.nodes()[0].contentDocument:document,G=F.getElementById(s);K=G.parentElement.offsetWidth,K===void 0&&(K=1200),a.useWidth!==void 0&&(K=a.useWidth);const E=c.db.getTasks();let P=[];for(const y of E)P.push(y.type);P=ve(P);const X={};let N=2*a.topPadding;if(c.db.getDisplayMode()==="compact"||a.displayMode==="compact"){const y={};for(const T of E)y[T.section]===void 0?y[T.section]=[T]:y[T.section].push(T);let p=0;for(const T of Object.keys(y)){const v=jt(y[T],p)+1;p+=v,N+=v*(a.barHeight+a.barGap),X[T]=v}}else{N+=E.length*(a.barHeight+a.barGap);for(const y of P)X[y]=E.filter(p=>p.type===y).length}G.setAttribute("viewBox","0 0 "+K+" "+N);const U=C.select(`[id="${s}"]`),w=(0,b.Xf)().domain([(0,b.VV$)(E,function(y){return y.startTime}),(0,b.Fp7)(E,function(y){return y.endTime})]).rangeRound([0,K-a.leftPadding-a.rightPadding]);function de(y,p){const T=y.startTime,v=p.startTime;let h=0;return T>v?h=1:T<v&&(h=-1),h}(0,n.eW)(de,"taskCompare"),E.sort(de),fe(E,K,N),(0,M.v2)(U,N,K,a.useMaxWidth),U.append("text").text(c.db.getDiagramTitle()).attr("x",K/2).attr("y",a.titleTopMargin).attr("class","titleText");function fe(y,p,T){const v=a.barHeight,h=v+a.barGap,o=a.topPadding,l=a.leftPadding,d=(0,b.BYU)().domain([0,P.length]).range(["#00B9FA","#F95002"]).interpolate(b.JHv);he(h,o,l,p,T,y,c.db.getExcludes(),c.db.getIncludes()),me(l,o,p,T),ke(y,h,o,l,v,d,p,T),ye(h,o,l,v,d),ge(l,o,p,T)}(0,n.eW)(fe,"makeGantt");function ke(y,p,T,v,h,o,l){y.sort((t,k)=>t.vert===k.vert?0:t.vert?1:-1);const u=[...new Set(y.map(t=>t.order))].map(t=>y.find(k=>k.order===t));U.append("g").selectAll("rect").data(u).enter().append("rect").attr("x",0).attr("y",function(t,k){return k=t.order,k*p+T-2}).attr("width",function(){return l-a.rightPadding/2}).attr("height",p).attr("class",function(t){for(const[k,D]of P.entries())if(t.type===D)return"section section"+k%a.numberSectionStyles;return"section section0"}).enter();const m=U.append("g").selectAll("rect").data(y).enter(),i=c.db.getLinks();if(m.append("rect").attr("id",function(t){return t.id}).attr("rx",3).attr("ry",3).attr("x",function(t){return t.milestone?w(t.startTime)+v+.5*(w(t.endTime)-w(t.startTime))-.5*h:w(t.startTime)+v}).attr("y",function(t,k){return k=t.order,t.vert?a.gridLineStartPadding:k*p+T}).attr("width",function(t){return t.milestone?h:t.vert?.08*h:w(t.renderEndTime||t.endTime)-w(t.startTime)}).attr("height",function(t){return t.vert?E.length*(a.barHeight+a.barGap)+a.barHeight*2:h}).attr("transform-origin",function(t,k){return k=t.order,(w(t.startTime)+v+.5*(w(t.endTime)-w(t.startTime))).toString()+"px "+(k*p+T+.5*h).toString()+"px"}).attr("class",function(t){const k="task";let D="";t.classes.length>0&&(D=t.classes.join(" "));let x=0;for(const[I,S]of P.entries())t.type===S&&(x=I%a.numberSectionStyles);let _="";return t.active?t.crit?_+=" activeCrit":_=" active":t.done?t.crit?_=" doneCrit":_=" done":t.crit&&(_+=" crit"),_.length===0&&(_=" task"),t.milestone&&(_=" milestone "+_),t.vert&&(_=" vert "+_),_+=x,_+=" "+D,k+_}),m.append("text").attr("id",function(t){return t.id+"-text"}).text(function(t){return t.task}).attr("font-size",a.fontSize).attr("x",function(t){let k=w(t.startTime),D=w(t.renderEndTime||t.endTime);if(t.milestone&&(k+=.5*(w(t.endTime)-w(t.startTime))-.5*h,D=k+h),t.vert)return w(t.startTime)+v;const x=this.getBBox().width;return x>D-k?D+x+1.5*a.leftPadding>l?k+v-5:D+v+5:(D-k)/2+k+v}).attr("y",function(t,k){return t.vert?a.gridLineStartPadding+E.length*(a.barHeight+a.barGap)+60:(k=t.order,k*p+a.barHeight/2+(a.fontSize/2-2)+T)}).attr("text-height",h).attr("class",function(t){const k=w(t.startTime);let D=w(t.endTime);t.milestone&&(D=k+h);const x=this.getBBox().width;let _="";t.classes.length>0&&(_=t.classes.join(" "));let I=0;for(const[A,Z]of P.entries())t.type===Z&&(I=A%a.numberSectionStyles);let S="";return t.active&&(t.crit?S="activeCritText"+I:S="activeText"+I),t.done?t.crit?S=S+" doneCritText"+I:S=S+" doneText"+I:t.crit&&(S=S+" critText"+I),t.milestone&&(S+=" milestoneText"),t.vert&&(S+=" vertText"),x>D-k?D+x+1.5*a.leftPadding>l?_+" taskTextOutsideLeft taskTextOutside"+I+" "+S:_+" taskTextOutsideRight taskTextOutside"+I+" "+S+" width-"+x:_+" taskText taskText"+I+" "+S+" width-"+x}),(0,M.nV)().securityLevel==="sandbox"){let t;t=(0,b.Ys)("#i"+s);const k=t.nodes()[0].contentDocument;m.filter(function(D){return i.has(D.id)}).each(function(D){var x=k.querySelector("#"+D.id),_=k.querySelector("#"+D.id+"-text");const I=x.parentNode;var S=k.createElement("a");S.setAttribute("xlink:href",i.get(D.id)),S.setAttribute("target","_top"),I.appendChild(S),S.appendChild(x),S.appendChild(_)})}}(0,n.eW)(ke,"drawRects");function he(y,p,T,v,h,o,l,d){if(l.length===0&&d.length===0)return;let u,m;for(const{startTime:x,endTime:_}of o)(u===void 0||x<u)&&(u=x),(m===void 0||_>m)&&(m=_);if(!u||!m)return;if(O(m).diff(O(u),"year")>5){n.cM.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}const i=c.db.getDateFormat(),W=[];let t=null,k=O(u);for(;k.valueOf()<=m;)c.db.isInvalidDate(k,i,l,d)?t?t.end=k:t={start:k,end:k}:t&&(W.push(t),t=null),k=k.add(1,"d");U.append("g").selectAll("rect").data(W).enter().append("rect").attr("id",x=>"exclude-"+x.start.format("YYYY-MM-DD")).attr("x",x=>w(x.start.startOf("day"))+T).attr("y",a.gridLineStartPadding).attr("width",x=>w(x.end.endOf("day"))-w(x.start.startOf("day"))).attr("height",h-p-a.gridLineStartPadding).attr("transform-origin",function(x,_){return(w(x.start)+T+.5*(w(x.end)-w(x.start))).toString()+"px "+(_*y+.5*h).toString()+"px"}).attr("class","exclude-range")}(0,n.eW)(he,"drawExcludeDays");function ae(y,p,T,v){if(T<=0||y>p)return 1/0;const h=p-y,o=O.duration({[v!=null?v:"day"]:T}).asMilliseconds();return o<=0?1/0:Math.ceil(h/o)}(0,n.eW)(ae,"getEstimatedTickCount");function me(y,p,T,v){var i;const h=c.db.getDateFormat(),o=c.db.getAxisFormat();let l;o?l=o:h==="D"?l="%d":l=(i=a.axisFormat)!=null?i:"%Y-%m-%d";let d=(0,b.LLu)(w).tickSize(-v+p+a.gridLineStartPadding).tickFormat((0,b.i$Z)(l));const m=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(c.db.getTickInterval()||a.tickInterval);if(m!==null){const W=parseInt(m[1],10);if(isNaN(W)||W<=0)n.cM.warn(`Invalid tick interval value: "${m[1]}". Skipping custom tick interval.`);else{const t=m[2],k=c.db.getWeekday()||a.weekday,D=w.domain(),x=D[0],_=D[1],I=ae(x,_,W,t);if(I>Ye)n.cM.warn(`The tick interval "${W}${t}" would generate ${I} ticks, which exceeds the maximum allowed (${Ye}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(t){case"millisecond":d.ticks(b.U8T.every(W));break;case"second":d.ticks(b.S1K.every(W));break;case"minute":d.ticks(b.Z_i.every(W));break;case"hour":d.ticks(b.WQD.every(W));break;case"day":d.ticks(b.rr1.every(W));break;case"week":d.ticks(st[k].every(W));break;case"month":d.ticks(b.F0B.every(W));break}}}if(U.append("g").attr("class","grid").attr("transform","translate("+y+", "+(v-50)+")").call(d).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),c.db.topAxisEnabled()||a.topAxis){let W=(0,b.F5q)(w).tickSize(-v+p+a.gridLineStartPadding).tickFormat((0,b.i$Z)(l));if(m!==null){const t=parseInt(m[1],10);if(isNaN(t)||t<=0)n.cM.warn(`Invalid tick interval value: "${m[1]}". Skipping custom tick interval.`);else{const k=m[2],D=c.db.getWeekday()||a.weekday,x=w.domain(),_=x[0],I=x[1];if(ae(_,I,t,k)<=Ye)switch(k){case"millisecond":W.ticks(b.U8T.every(t));break;case"second":W.ticks(b.S1K.every(t));break;case"minute":W.ticks(b.Z_i.every(t));break;case"hour":W.ticks(b.WQD.every(t));break;case"day":W.ticks(b.rr1.every(t));break;case"week":W.ticks(st[D].every(t));break;case"month":W.ticks(b.F0B.every(t));break}}}U.append("g").attr("class","grid").attr("transform","translate("+y+", "+p+")").call(W).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}(0,n.eW)(me,"makeGrid");function ye(y,p){let T=0;const v=Object.keys(X).map(h=>[h,X[h]]);U.append("g").selectAll("text").data(v).enter().append(function(h){const o=h[0].split(M.SY.lineBreakRegex),l=-(o.length-1)/2,d=F.createElementNS("http://www.w3.org/2000/svg","text");d.setAttribute("dy",l+"em");for(const[u,m]of o.entries()){const i=F.createElementNS("http://www.w3.org/2000/svg","tspan");i.setAttribute("alignment-baseline","central"),i.setAttribute("x","10"),u>0&&i.setAttribute("dy","1em"),i.textContent=m,d.appendChild(i)}return d}).attr("x",10).attr("y",function(h,o){if(o>0)for(let l=0;l<o;l++)return T+=v[o-1][1],h[1]*y/2+T*y+p;else return h[1]*y/2+p}).attr("font-size",a.sectionFontSize).attr("class",function(h){for(const[o,l]of P.entries())if(h[0]===l)return"sectionTitle sectionTitle"+o%a.numberSectionStyles;return"sectionTitle"})}(0,n.eW)(ye,"vertLabels");function ge(y,p,T,v){const h=c.db.getTodayMarker();if(h==="off")return;const o=U.append("g").attr("class","today"),l=new Date,d=o.append("line");d.attr("x1",w(l)+y).attr("x2",w(l)+y).attr("y1",a.titleTopMargin).attr("y2",v-a.titleTopMargin).attr("class","today"),h!==""&&d.attr("style",h.replace(/,/g,";"))}(0,n.eW)(ge,"drawToday");function ve(y){const p={},T=[];for(let v=0,h=y.length;v<h;++v)Object.prototype.hasOwnProperty.call(p,y[v])||(p[y[v]]=!0,T.push(y[v]));return T}(0,n.eW)(ve,"checkUnique")},"draw"),Kt={setConf:Ut,draw:zt},Gt=(0,n.eW)(e=>`
  .mermaid-main-font {
        font-family: ${e.fontFamily};
  }

  .exclude-range {
    fill: ${e.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${e.sectionBkgColor};
  }

  .section2 {
    fill: ${e.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${e.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${e.titleColor};
  }

  .sectionTitle1 {
    fill: ${e.titleColor};
  }

  .sectionTitle2 {
    fill: ${e.titleColor};
  }

  .sectionTitle3 {
    fill: ${e.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${e.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${e.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${e.fontFamily};
    fill: ${e.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${e.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${e.taskTextDarkColor};
    text-anchor: start;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${e.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${e.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${e.taskBkgColor};
    stroke: ${e.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${e.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${e.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${e.activeTaskBkgColor};
    stroke: ${e.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${e.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${e.doneTaskBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  /* Done task text displayed outside the bar sits against the diagram background,
     not against the done-task bar, so it must use the outside/contrast color. */
  .doneText0.taskTextOutsideLeft,
  .doneText0.taskTextOutsideRight,
  .doneText1.taskTextOutsideLeft,
  .doneText1.taskTextOutsideRight,
  .doneText2.taskTextOutsideLeft,
  .doneText2.taskTextOutsideRight,
  .doneText3.taskTextOutsideLeft,
  .doneText3.taskTextOutsideRight {
    fill: ${e.taskTextOutsideColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  /* Done-crit task text outside the bar \u2014 same reasoning as doneText above. */
  .doneCritText0.taskTextOutsideLeft,
  .doneCritText0.taskTextOutsideRight,
  .doneCritText1.taskTextOutsideLeft,
  .doneCritText1.taskTextOutsideRight,
  .doneCritText2.taskTextOutsideLeft,
  .doneCritText2.taskTextOutsideRight,
  .doneCritText3.taskTextOutsideLeft,
  .doneCritText3.taskTextOutsideRight {
    fill: ${e.taskTextOutsideColor} !important;
  }

  .vert {
    stroke: ${e.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${e.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${e.titleColor||e.textColor};
    font-family: ${e.fontFamily};
  }
`,"getStyles"),Xt=Gt,Zt={parser:De,db:Nt,renderer:Kt,styles:Xt}}}]);
