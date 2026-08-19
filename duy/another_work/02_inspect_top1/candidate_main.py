"""Replay-derived top-1 route agent for Kaggriculture."""

import base64
import copy
import json
import zlib


# BEGIN GENERATED ROUTES
_PAYLOAD = json.loads(
    zlib.decompress(base64.b85decode('c-rl~U2|MVk|g?H@C=QiJ^|=W-Kw3Ity+}n<Cg7?!DL!B$EIi1<TX)!+h*JJ-!Dl3g;OUzJlrF*08q;(f`vj=p3FRXB0N0Y|9SSme);j$Pyg}q&Dp2_Jo{;J@yVO@CvSfJ<;Q=lm!JIY<<GBQEtc<o^W#sy{`K{%w{QBlf1dsOkH7rw<u7l4Jp04<Uwrxb4_|-w?RVdQvpc&vKl_g#U;X9Xo8JETw;zA`-!FfC`}4n@{ppMMztw*D!}tH=htIzH;?JLbd3JvG#~1sv^JVk$w}0C0zMlNz+uiQ-w=eJiwEOJ4v-8W@%YWW|`Qody^TpxqUw`xB^Y8!o-NRpOZ*G1&pI+G4U;OdEfB*HvH!UXD_HSqV-M8PpAL~~yzWMG??|(YHH2dLrBEH>y`Q^j6uIAr*|8bMA0t1<T?ZZ#=r9cCieO<VE_!|yGdA}%lAmrP={_?Xw?u;3|fB(a`6)y4abEgy94CDQ4Up?%GqieleD*I}8pUuDcVPkyu-R_(AV&9M8cB_Ih9q%K!iI?;3^zDo9o9)zm&tX{q?cL3CJifc1*rK-cG`(bSrGNbFZta}eY?tvc>~`($FB@HL{#EE6Y$xq-Vc@2i{4!p{*|*+JKAgMgXpt7^Zo1*EJ)OhZtwv8Vxce42m5FRuk1TBAZg2MTIlK||g#uS~_`JJ&p5A4`Ej9aNI)8_+Y_H|vO?Nj0o4C8bhBNo@9qq*BAN}^Y{`UBj+5HJ#x&8b5AFOX;H|}J@I=&(Nl-`kNpY}Wa*x93G6@-(?$!Q!9V?K3$!a9e?qmN)7mXNnL(ZiS#{PD$?U+(_+-4Fk5_sw@-eEG%yJU({@O<ws+j4e|BjuWlH{X={*p70(zWFoUS2haA+jS==e>-}%6@BA9q_4GCm_S*Ev36I+#uMHI^5jfbmmGlCHLhzozSCvN&?KJUgm|ksmjg=^Q{!)J+ItxjQbAf(JKM-sgKEgI`?MOr8Axrz6W=mn5-+2Qy$!slOEsc-vCjog%KkgwW7Z*QIASPjc6Ab1ze~X`x*fJlV^>JyaN`SLR4y-?Z+W1eCpZg$&T4Mxz(G?UR3`Pt!*n_R_5yya#;OVSJ2iD1-erP9bO&$%SV+6W1b3CT_cx3qcNunq4&H96ih7RnJ+8IND<uo3KESl(=4iZ)*5AdFVHGF(mBLi|e&x3|hr~dqEK7w%{KlXR_u{s_VM=o}u9#qnUl6(?nZEQTG^S~5=kz%@+MkVnTde5vR<QjzykXh@iiZ_5zIR#}8^k8=W^w~H6kDom&(}OOiRC@+H=HauC#@{tT4?i7f0N`tiF(4@ZPv)nPXb?Qn!&LxFqb+*^VuUgyVBFL70+TZRbsFT)%j2NzIjNiz%E^TEl09N*nHXQ(;J~Y;0DX)SK7MR3#Itb%rOo@(=7`95rK3J!By|@ref!-vpY8v!`{tYf0r-JZ;fUqH;lRPSFK#Lzxe6i7M>3?kG=D^%D3t1kM~q-?j$U;a9gJ|qBc%<3(n5f$3=@HCFYONo59J@I67k+YlPC8fW}_By#Zmj<5E~tS=z~<-&kJVab|}?o5wPWM$5JV4#&omUm?t~>{s|?pr5V*M*d=;7j67Gz@Zvs}nx4LJwY8&~M~YQ6?BnRG-rlC)yg)4z<};<3xaq``%991c5bbkt|Ki1$?^{)17<Ko{yA^tSLVPY10}ltHSFtNIh|b}nHuN&aPZ&fPnNMo)aMZ)t`^D+np81v7q0UKas`JNwZ!v5L&qx7#XMsj<02n=q!OPwCQ;Uv6%6VWF)5$=<PuL9LPXHV4!-sZO9UuPuaNRAMR3Rpj1f%$gZ8%(LeIt~oQpSw>dD@_{9q8<pM<Z!<g+tp)C#qdNgQyFvpRM~_MKgS(gJnb`Cb@$Zg+AzS3uIZ;^!adzw*(<C96~*S9n~Z>H=qYgXx<B>>gHj(5=gt*2Abb+OR2hA0g?(43+irVUEE!bm9*OxJ&A=xC~vsbfFG6rHyS65+lwPwRAC9)+vKnS!fF>8X*9H=yl?rCAP5|aIofcS6xaKKy7}<~I!D`=@|Y8~?xSeCo$Eu8@Gwe0eqg)$;d7Brz>u$R=N{4TcF7l5UrFMxwitfuZ+y7O!!7)17pZ^Iu!|h~l^_B-{4LVI82aVl#s$|)Z=4f>s)wCaSm~k&hL&mc^X?WGIB{qKJFpwuL<x!-ptr$rMZcPKY=`3pg)N6)8uio?Ar94|5osq%8$!twIW6mp3|aSJ`}E|K1;J`LrA!}k56_pobL!7ueEHw+He2v5AhL`y6!|mvF%Ro-7vFx$VwvBZOP27rJPS(XoouII><^%RbKIGPjbdh^a5=zkV>^)~2*vgr7)&PFTnlc2y(f0@I4tHC%%tZcVp%fvn-UipaGnpTR#vIzK(Qowl%=TZ)+DW*hpz-QTRoxW9X|zzYCCxk3{J2IVunVxSCv|+By*PKZP6xhXH!>kqU8ZlD9968{HCsjK5VdTqR3FBQC7-7;Ea(;Jj&=SE;ImT{J}^|J58fz_nyqyj-EIatVy{=Y^-gWrn2ORk7+Rgott#(j2gt+Y=y_<e}szPep*Yr54O_y#9V2`gbHavt3Y{RQl?rXe;bNy{umcG!;6d(liVyX8nwa)kpp<ku=kB_DNK{rhlr~USTihuP(hLu#}!xrCs2k9fT36&kbNQW!=o0aKcq=cM=kUVG0FAd(ci{RDR5BeqC5?U|CpAx&O$C!x+Vp$3^2%(a>9LE6Zj*|9h-8}<THyT#UvvaQVM#p`5D`8F9nCOq%&%@!*+}09D-oV(8)DMfD17`**t*X(S&ROM?y*2lzBN%^vlBgvj|zUfS5cdwJzrZu@dB_WC=k#1#can)%QB<-6%*Rv8NP{B9OUylG40KiG-Epqmq$#WnXfwRPGsL&mcc>I3Fbi4JNLuNPN1Aa#x}?YFLCN@tC7hZ`T6D^${lVd5((ljh}E>dd7dseEm7q=xK#P&dZG)D`qO^6A1z4EI()Es9jGdtcP(3Ll*PyveHo(+)-g^&?HEY-LD*eGYgApJdX>7RcM23j^vB*mT|JD+7<wbL7JS?(zy87#cp`#e!?@3UnTpvw%jB^!UIe;B$+-4<SwpSq<Uc>p)`vli6*EmwwI}lUhfp-i90zM=u;4OHHs6x&4F=LyG{UH&L0O?4A}tWA}}*e*flK!Y--;pEkRp6rsZj9v35`Lntpk8#lXXc#1`1oshtW?FODv)x&_hIbD>9zdq_K-&{l3>;$UwuI7(jH>v$Wl#fvA+&*`I$tR+&tqf7u{qaW>oi<^>IHDxd`q95Zl$s0Flom?7)#l%c-y5+Xfb;d%3nFP^%XKtf2XHnjMfTPD&1K)Y!Zd909+<GXvAO*BT-7x~^Ld1~O-MA7rqs1K&O9=3A%yHDxA*o)c#61~|be^8gF(HzZX7O7wmO%~T%T*OF;uw++9nE&e<*8U6xwrATqX~F&zQ@B;g!F@h6V-!D7+0h#8wNm*kRVe5-s}=%YmQFLtcpMdZ)o?Q_XTYb;asq?<y^WD*hT5sjRvF(12yK|QAD^nBJu`Wl7v3%RUzc<MjN5&FKfOdgy{iPy;;IQi9ivVqDb%H-70Z96m|=MesI=hQsd-HK05BQ*Sho>!ABAYU>N{!A==dm6{>Sh#w@4I8U8m#T48#spbSuvfgstWH&n3xiJ%s#5wNR)Nvu|Y(Y%b9$QT6;b~(g=SIs1~xF4V1T*<A+@q-c`r|iG{0@!ZWMaBqV7Jr`pOSXCJ?Z8wj^A5-M9B`-A(U@a31Mz4m?QnnyRyILtV(ECY#?t}&qaw0gr<gfL8yzr5zk<a`^y2X1bUarH*fEF*L-J4%DzHo~=P=i@LqyGRWGiF<!X!2%4JS3`3yEf}UefkdZ<p0<MA!`zS<Wz(T9!aFa0B<r;GPb;4oM``h6$^Cfg+y4Ik-0Z1KQ(2L~u$pGu8}X!!P6Irp1bxsQlrXq7nW9>EBV-gNxna!d||E+I1sUF0j*eK2}d7N1|WNC{u9_*-mxH+EQQyoM`C%nW<n+JjHJqBG5vn!wq-g^2U%?;N77K2?_sk1)O2T7gp9zlL!&wQP>IjRcLg*kk@Bob#R<lVkI7oz65-*LcEPUp?Q~-jjQO3n8+)mEKj=)cvv`FVS+QA6Lp+645;3UD~FtrAdd=%Va8FZOsSi(-*zcqMFgVDDaWr#?x$K5Pm|Y!-g%XlFqPapX>Q+2;!aOxh)irW=Cjb*Qq21Gt`VJA(M3Aop+GtuM~u=!0hA4psY(>#0I_joh!3T*XHcdOj%F1ns2QQys`Z+#r&(%aWPo%vwcJDCW!6BR79Rr?^lbW^zMe<Wd=N!5jOHdq(m~aGv{W)iEvh+5gNypyZ$O|!GonfH#uVu_EZNX-!iqpDmEutIj=UzE1FWo77TuPvTxP9o2?BGv#f&;3(Mv4*m@(5<Vosr^nz7!=i7eF5nSi0F@K_rfy=Wp;!YZV>Kqo2KDn=WzN+OXK=^2&8mq$|~=s8B3G#Q0{^mfZj<)^3lPfS9OlIH6uzy6>r;zt*~IC<6_yQz0}qj#3S2Q`^HX{n)*P>32CJwX<B9Ov}Vu2V6guC?2fWD`sVtIp6u{h(M3EE$NqXj!r=du*DrXYHa{oXzhsza%9Q%KiOf)jv(Q8n#MhJr;@k-G;h{?-|7CAt_;?Kq6+&6&5DoR!FiF(~$}!Mb)K+!T`q$5jSU;<PH@9P3Pa66P85sx;CrPvO{7b6RR+CAoDDcb0O6Ihf%`yD^keW(>Uo)HruT?Hzj{>QFLa$we0v@k!K;S7pVbUtQD@DnS!(NFdD{o$QDuQuvQu>{K~NSgjVH125^s`zC39rS%m#H<XV(azxCn84`w9hfUb5S@~I1HKpS;e`*tOj=sd_$`O4&nY_gpv_7Dm)s7uu;;ruZ$rh2LHm)o6l?D)4>n)9F5yXN6dY80qMDI}VfH@Z-cged@}WahVuS3hF{cX}yO3Op+E=(HG46FVaX?jua2uawh0oehFMsYa^CjL<rnpQe}AMs3%hlL0kR!?3X5sgO_WZ(0L7P6s<ITWQ%(5_}YTqG}aavDU^BgGBbTFg~8&Db3>`L1)G6nv0pmzvMr#-C8k1)eJ%z<<%ksjLc=F4tv(xPRSEGon9HD+jt-f<5ip3$zG*L2ybi7yd81Hxv*);qG~W%=;%mG=QgB3sq&q5UFP`Y4*V1-!&GzaGKk!l9~MLUq%7#^gD53JU(sj6xN>i;PnW8C7t6qzBzyOu$BD}q+nnXpw1;V^wXK#XlO~-)DfilGo#VL<iOwsXrOT4=Vy3AUl!D{rMDPqPsNZEVIy{56Mu%)s1qSIbu}T-wb?!6{;6bj`+}){Ac#J{V(@?0!1EM>nSLQ7*#<M^Z5QF8cc{DVE?%@=6Da2_Kb&456jUGT_bJi7+sJNDyHnjE~KE4%v!DqR!dOGR$M$+6-c;@y8U*4EJ2n`c!JF-QsZn@>NDlqN}_9>p8`FPi=thS}1CqXroF&2%sfF(&~dPhpJFizI^YquY(_(soq50>s)tb>gAgK7sxq2*)CdS%rN8j3DYS4;;rEE?4NSzKRUZ1`8v7cI)$?&-RJ3&iswnRsWGO`1mK9V19j`{|+7GaLoZl2JH|i*uMjVc(5uD?X+=Tx+FsnhYA`hNu%}GkuQSHk1f{W>P60G&Jj%OHiW<%%in3(<~;e+~AjHTe?v1W*|YD6bE;d*lSy;fU|^^_72O2o$$EmY=hnlE_7N!&6qr?FCiy0qbWqeE_}plQ%Fb{9P1D;Q?+7Qp$gHvR^n7r`i3Lz*iK<0z7{QawD{^zitHVNTjDz;;Y&07!%`W%%>_9w>M93vL_g<ew9Vxu2{^?e@SJq6*M`U{*4mrWzZ=l23`WA#6`NXiOAm-TzdL6U^hxi&b(_vl<a>Y2dffyF!(F9lplH{4O8@2aQ4(+Y`;^`yDT$wktZ!wwCB^A+jD7g5DOk$EqrFV{?T{eXiQuZ7`pJ0*WKZdx-#+1bXfJ#FNQ$7D<{DFoD+}WyVq#wX>Th87f6_Y3q9-<^?bGsVLfy?@?JLOlHK$8xL9EuqW_W#5q3+Y2kni0rVGEMSy0aAZLjx^QWOk#rnccUZTlJ=melBke^(%#jLscuXMR1G6cNeh=P1CPyG8Zv_FkKVU5mPTB78g#EWf0q1m|lf?u!h$53#n(G+4$MIW=Y?)T0~V=h7_i=P72#>j8(G}U4^*C+TVGyki60do@~9YrP3<k1n{EG9q3X%|KeZMhiGM`E`^n9uOkuKZAR^R9^5}BraAbPM7YV>3l~9-f7QEVC0NS$Q|u#Ysatn-Q!#|o7%xouuWpv0Uz?-*ClYBqRluDmslQa74KBiU1ZQ1QN&QAT0nGB#0ACNM$hba|G2EIgdGKuWMu1&e>judi@Sg@rU=Z_`oYxuoCdD2gt`)t(BdLs$GPEoL>y(^s--HrTWTBi|3isZ?8g$b~mM%MHL&5W=ueXM#zNz{htr&BbNxrB`nnHHGE=j^`<=Q!_MnzW*&&OKZ#lqOuuJEN_%kJ5}C6j0!%D~SN@Ja32&H#Z+7ORL$MmDXeZqEfrR7w0Xw;<Vol7I!W{$-4%F*$TW<Q9bI)4g-N09xOW1@e<q;lgb;0aKmvF9;k5W@dg2E53jEGO@CT(*bdw*p=*phPW=grUdk=yN^gQtBs8EN?nwWb5{jpuLz6|t}EzgxFx-r$fZg)Dm$}}c`YnbQvGL=Dy7RgJ|dZP?_+H|RmhesE^T(Tqxl3qGC7v0uSQr@8uE65p4Luepk{^3x+`>q7B5eO=xJAN3*m`%2g4(z7y+KN1#fj>5aFIoaAu8=z_q7&yY45*^&}l}v)~$c;_1JgfYl*gkRSgp2I&g<q5xP1sk&PFWIGS}_uw?psd*Ka4k7bPLg$j6|EYTR+MSYhy#b#Dqnpy0Y0~BaqIs}|;k)3)AxH#vCXC2k&;}JSf^<8_>33yUV@M)_nV?QIj&iB^PUjZRHYA()ET}R}rdWO+bSwqQf!^{<8(m7>EDT97h}uyDQP^&)3#7tuIeS<I^B&NPJl~m4-Bj9H+2T!_7EX-3aX)l{qgK=m$Cz_=REw?oMcEcQsfk#FXfc}juFiRp`jF%EpHF!Nn(xH-TKfl7--ug5b`NBYr6rc*+60EU2eDSdpb|Dpf$5M{)|iXQ))t%5DVFIXB^|6HlN^?rj<L{S3_I+yjw>*0C@L|ZCPmr<#>OqL>+8zMkJ%{(Za)GIg|5M40KP8H*b*S7+oxlKh1kc_h%6<2><ksoG>0FD7bf;gnJs&jj+o@RClH`g`M~H6Nzm0Tw#vM0>ohRSm?`7ou;`F8MYkoYP|N?KjciKx$0~OPK2(Bo$2~+>j#ZMXR28MML57Kl^vhtn0_Z8@`ZXV&>;X+|Hqp~CdTjNGLu68|t*Z5faSv4DCEhNFZlL2GVyybM#6$wki&?{%EM?LmT6)p9isW$Dv<owqTt?qE*6phAWM<j1zyb-c3WdzYO&G4*(l~>;O+*msvns>1X^W%H!n(ygRlQN3EGbyXcI@HP<UYH7`vu>;=uqZ0o+|p1g9lDK5$Mvk(JpSDvGa=;yGo0DA<@2_=e2j^Iv2{(+)Zw0=;dXyVPanE-R&#&$~#Oe6ZW!1%w+m_L=xN&tdzP;@YDU>pV)_w#$wQ-=GR&1!)j<pliPQ%RbXU#(x!Ly0(7Zq3D)?Xp&RMws7bybC9k|AlN7dSJLS%cIVWQRVacejtks_N@;A@2cM)vb7J3miRxD1l><mRtu$TdUeghk4`nqwq3|i>wz10uZNB4IdDW+$6zrifDVEk6UurbQ`UB+-S4=pdvyL$N+bXgRH(*)<NlIBx-&=DSWZwv38@a08clNs2jRNm27{&V&yB_#K^>l%g+TB1angT@s8q@Hv`i%@FL5Z&3zwFIS?JTLB0FI$}e*Rd;C27w$rr2(tZ<Ow8QCxw>PPM9jWmsBgjM5)UFU##%=%4wJs1-gs(nFGGHk}&Ji$pTRk9+78l;+fc)n!NTcBcT8@9#xx`P23Xs+R-jRX@$Ap$)ctSz>Gq5b;FGE+BS+jb{iOl@<{nKv-SX9Fz;v7+rW2$TeYtbjuz}tPkkRRWoUjdrh`_*1L@+i&U!F>w9Si;MPNBprW*=Gj|9GEjBkX$p-mvWW0kZ9h&(EOjwl|T<{durBw7@?vpmR?1!Zz3Sg|-ET7ohklSEM_*|!=DY^>?}G!#-%<)|skN}cBN8lD{GT!HC3+Br>5^q~+-TXx)sM>WmyqDLg2?xnI_crCg$UR{9jbj(wcuGXtSxPNT<LF;N|_%eE~xZqI5U6yt#Xo}z|U|CTS6#eH=h!7Hd^wrB73EJnt%3$*_;Wm2f84O~557-&aRUfj+4@s>{Dg#!;D_M;uP+G0#EMeo#kpx_PM6|tqOsCL^4xRPv#U3gp^Nr9b7)%H*^*pVUHHh?}EyQQ_5@-B;YYk7!nQSZRY;DusEJ$XZoY~yiQA_Mkdd$M@;+OTV3v|zl^D|H?WSD)X_(e}%=dQi%#r>G!s;rEjlR3NkiAxfPK&q88#d1yBj6At7GzFE`O!S*eC}PNB>YAF0NQ)X-g}7zwq`i?DH+o8fFhO3W;X|cm6RhezN_GFQ<J%?4kD9x4QwQ%vT|`CHG4}A0SU+9dmz<xP$Pg>)qO@c^M&)~TD(S<JlX;8mT|R*m$_fFID^KBYl}z_tA#oJ5blD?%xS<HF@Y_W6@JjSud0tY7MGN8aB8QpqjpKT}7i{`E6%EB1u0g4pa$(rN9qJe|@pns_P7^F78s**7s3s)DA5Rif*C>gJE1o0o9Q6R|z`CZMcY=Lt&>AyvO3f2`EZQ?77_VzrCz28@Qw?>E^Gp>Utty8$S~NeTi8(q2M0DJ;$kuiJVl3rcZgWhEZ`R0JCC2BAadK;+;1YE)aDQB($DPxHgx+{zve!DIrRZ4#oJ8$1Q^fzqw0~x7E-c)ditn1j1>F$cKt(kb!Q%xI8?d|VUo4ebr#ktTmgl;YUYF7rz}+O!lxug8!W9&GbGb&Oz+DFmhS0_9G6TQ;lu?qDtu{%k^FfnTgy{L4!F8Q8*e7GSmaT861@8M6X0R65AMJ1-%MSPHTySYt_hDMyS_HZe&g9luFx6agoR_T*{UwosA->sLo9uS9mHI+-zE#bn_(EyHxEkqj+Y#A%^^*0;3~M<FLM0!ojF#(CQyOoA;Fl7IuM$;<u&uxri%2?6YfQh9?R4~Qke75<D1u&nEtg5`D(z9y%A*m>+Z|gvSX`Yl!}?aU9>$N|rdPyHhZo!W3NjKIrLRWFBk4hOt)x5TjTQ#Oj7W7!<j0Vh=AWxaD``py>h~yB>uSjo6a6l_Z@KS9nM)8Wf-E*SwYm_^qC<;ID7e?T#;F`++3r-a#gPR>w^;{$Na$i<L$tx`I)sY>;AS&5;69KXf3s4pV`c8IKbFuU;2*Ep#y=RXT<gwlt_UDIPBZG^KisRuIiy*C1iNa(z?he(;%u|LIeCGQa%D4qiyr`Y*a`LV*H&20<=IIG>O8h>zf*5dw=IiYZG)mRoY|6L6ga5BIm>x6lJK7eThR$sjSv{e@urJ5Q!t`5S_<>wZ*ri@+CmvhNgvhite8w2mhu9>j}}l=wUa1QysY5GTB<dVI^EU86?0ryTA(yh9GN4YpC8OAgwerj#(%@itch-&DDp^kALAWt`;0PJf}^)4lS!LuYb%J_`ZAg+8=;mGJR$_g`BEyc*|@C`rH-`a@(hkHd#JR-t!1lOv@FL^p#HID)>PfOhJ><GNzxW2pT>D*qCvTCQ&rd%lr`C1r2?YV5kytyEbxC+K~#XoKnyFy@LVggA~R6)h)+hXbLd2TRR>?JW;A*#DS;^$igP%19ZiJH4#E`VwOS9OUv;778FF!JUOUADV$a)|)9v&XXj$F<vO{1e*q^JhSPW1pIN}61g~Sg``E1<ypF;m^;ZzF4k?)1}JHz^%-Tw}Npyju$MbpA<ikfG{6-?=0V7a&{+BQPdEWHS3NC!(RMHdY61RSKhwlVa;Sx&DbP+91sguEjpzcVG2y{yAaQH&fFta(vUuK?2DZxK(01ZvUMNG@B;R+`F8rFmV<jo5Lb=sRh8p+Y>?NI}X?{6aNp|6Jd~8Y0W1TYak5%uwIQl9*aS0gf?a+gM$f(|1yuF5N+c!a*Bz_o=0u#p~!!$7J6MW(dYDxbw^_el{)CiZ}sq81MN04#``aU8Y?DuOVGWlyAYVG-je(qn6Pa6>#wdS#L?~;_Qe1j;>I<Qtjn53*ijOm4&(v1=N;PysLO=j}~@Vpruj%dTy!aju-1``xG+g%?XC%D<vZdtTAw9=>Qk~d&CBbZUgvbWP2W|%pKpxu`hfN5Yyi7r*#8;9P7^!08GCM6Ssaal*sA*2lw*&rl7(k_dyE354$ru#ci1Oa|$k)Er#vE+@;7Q!v8D)6IQ-~wBgms`yjoO_#-fd$o9M39nfy#m31$jk~*Sfi@vaX<t%c~o1`O%9Y>jDW4iHaiv$mOH4$3;Yi~pubRv2sGyapMBm{<-Z(3ami$GhwQcYa=Dt+pTPPx`7;Vq$@MG5OEy@p=#sC>x{RqF1=W;h$AZp7tTDnK-7jOv|m7-fd_9Nsu+fYPn7o7chTqnINNJ%a?dwNk7n<@>gdUl~`8LtqUn2X;?3s^NQW)W8C^S98BugPRu?N<h_iAq)3h(){?In{-u4Hxu=CLPRk0-AYrT0+P_uqH7$B3nr>)e?=o$lm_OMLnUxTPnVt22FQ)pZ||gHS4~0-&7+m{+u2pikI(3M@iHtFT(+MKPbvL-lW>t8lpWvN9@ffb^v-eZ5z#rpqn=A2);H$sbPSV@_|7)IQ7D!l>{XIWIn`ByO95cZsV|eHfi&l<HBPfDB_`|Si$DM7#n&^dnA=zSey9{!kR$Q(=GmPm$8UZ6zc;V{{>x7<&pti7UM*M4i>vE@`Oi16U%kDEf4qG8mmglg`uV@kKK<v}db4=<Y41Me)9t&@TW^+!7uN4y*j%m-FI=46e&MH!_rJNmKKy2R|C`Hqzu7Jpli$35;o(E>-m#g#V|#dEbNj;O{7<fqN48yD&pu<hnElE6{l(nP+WQw)({K6hxAU`~wx7INEI)bk>n}h4W4-+3Z!dp-{c5p%H#a~2^y^<=zj`;h(~1AX_g{SZ`44ZW|GV#x0sLyM>aK41*;ilu`Lp+_$BX@hMSpnt+du7gUx)Pid4>MZyDwjSb#J!?vHopXnY{a^MaH(UY?h)510$n93<RGe^Q91~Z{h0UZ#WF){h};f`;r5NL7dV1_dk4F;S%@AhH5qp*dH>h!`fHNd^!Mfdq0BPtqR5@6s#0W%JzHq%y1{IUx0anc>%Dj>$ASuD=nDq)#hJ??!k7_9ul7RrU)CctctDX=~k$$VjAV_R`VqQlqyZS%iTV#l0#eG>t~k78Z?RAWx_2r`(rwPSO9PKrcw*EW=NcWG|hs-{V;i@CyymJZirb}-yk6$C(XcrKOft=@C5-qJvoizVa%t_Pgv*hc=Qpl!a^23j2QuE(F~fr@|PG}r2HKxS_337u*zyk&?dBbwr_5Xu<u##e`9^;*SM~yw|TI)%dsLW$YVU|q9KFgrKU=bNs$BVv8|9Pd!uB7*D400vyikn7cuF|+cJEFh+0%mLn%+J!f#dL{LUMQwhETCEA;Wv{Ujhyh8cjk_;~`siXHjQ-{L1Z2V>O7rJ*VT&K^0i{`hI*KTUq_Garg$1bfjH6gYGk<H6SVh+{yA*B;<RIzXW>`_N9<nmigt#|R>OI&B4Dk4J{DpCo$n^hKS}fjv??V+dxo(HBj0O$P}pk_UKCz$%b2y~x1e>x??}4L;dNFz(~Weuy8d<56+s(l2(ujY{%Kl(n((kj?|s)>N{-rXtT&dS<y|??r5Y%vxVn9IAelQc(6l7u3k~r?4_TM4r0b4al3F`X&hUVtVXvEztmebd^Md;E5it0$3Vt*%J^Wij9<k)D}`ZOoM!)T)}i+TFS|U^im+5<MtOQ%rfG+*iwK#MhPE3win{rIDrz7G``YNpD>cTi<hQs+f}J><k(4Fk#BcgvHDG<hg6pg(f^)Msv90Lg0(q%)m?Nj!V!;@HV8@!0je@g6gM`HgNG{SEkw_7avx$gYSm?G&^}<D6qS7dx;OAl+zzE0EdsXO?N};h&6sXB8#Cw)5!lj<Y8LDgy&OiKt7CX^A4^S7U%1-ZQOzU8DjN0y?J<@&eb1C);-(W%Do>U~wdejhc-xr24Z^6qU*4_I+Y{n*p%|Dcew9VfgCi7so@{Rne!?Ka$b3?Rhoc_G-Y-tq_RO!u4s}jaQ#B;Fq+-|*o{<9f>b?n502n=q!OPwCQ;Uv6%6VWFb8$JMJqqQ5yb0J#0#uz<N1(#gqDd8E5?4;Dd}1387h2y4<*Af0qkf(?Xlw^Md*#tcT3zALw$h1eSI<BkwZ`?cb$_d<CPZ3d8PSMI?qFs1L({)4kY!QRXKUW{0NQ!s5b6Q!s3xJgAworj$D<n@t!@lmn`|1U;hr8jcZG9l#KPREHKF^yf|sf4Sy?g-_)+<PqjAExy;v8s)bp9v^<7x)A|s84R+RTG9})zCqbm8-;k9dmr$F8O_>ruf3xdW$v`w^((;tev`r+!4PQZ|_SLS}~1%vP_nebLjeC+q5y-ggJvf)-{;1F&mm}^csRAN>#0Hf2``brQi>EdZcrrKv=*Tkv5k|C`~5M=C^i=E9Q>_KCvK33_~9Bshvq^_vpDR}}{3t};kBA!fWUKL!gu$Lg;C(1aO1%_-bQ3>H#EK?Anmy=H24DkNOok-gT-vVOKD2I_hb1zuN-+szsncto(a(k){ZOkM!GZz8IiXOM4n8YlHl#SB-N=W)T37<JP#WhLjqB(=Zk6HFmahyRNu-F+91y!2#5x14-*n(B#d5OR*8yy-}u7M_hCZv?zQnTp3d`Qo}96;Gh=lOX8uoQxOU@fF4GUvTD6$SR!G8%Q1?aU7uWO4S0Q%C23z-sstnVV&vcx|O=MUaG7dmF#PEESe%?MJg>_xRb#I{3jXVUjjSbNzJFiCV%-ws`xBHLS7BCqb*NJRB+C?XMtaH1S&X&?p5yXc*vQk-Tqo$Dr;}2%-|%7;e9_n5n*g0tjQX++8!vbTx6*aMmzj+1sX-9!}zO=Rr=}g3f8yUucr7Q1lN7qM*`t%NJYYtu;YGQ30IQw73PQWClt2ZR-%gSQ<X5%^Z}8h`jX1%ZvH7=I3m?#T5KRyj6P2gY6cf6bOlLcCLANTX`blb+W0T9!_3!idt4D$TB+EDUw1obwiZ32rjb_lRPa&T%%AuRWb&8KPx^_qbjRLGDBKviP_2+V2WhA##qz{I~uu<OH+z~lmH3dPtA#EkDNWjI?V!mk|!{sU3Jsbm6FFKZl^0ZKjx^^yS%`HeS}GTo}*%%<R=`K4jh~^>3>eecxr|vC-+987PFc2iG&b!mgFm?s#3SC2b2gy7W0s@8d4W#QlVo|Q9V8Kzq%<onlgF>h^7M{`XW9;Hb?SBc*{7sS8WS`v_i)$prMfV*8?16v-=6pIDVDv<Jxi)b|`oL4fY9{K8OY{u7Ee70jJWcD#?;TTUUFT%II|#08jgs9A8|0d7`&B23-JQE^udXjybquZg$>Y1U8Zqc1_Cxn_B!yOF+ucB%Q`Iuj!XpSB$Z@B)kSr1prU-3b(ff(baP?$i+>ky8*Z=jD)bg!N4qeX|LmLycRE>u*=d%8(B-lG74!b;zmE(0~a?XL2C-<X$>?T8&{<|lLu{LI5^#M+vqw0ON+dX?aXa-<}AwFkI|yQa3u_q^v<grZY5q2=o?}J);og?f!0VOXE+zs(h!K!BO^_)pbnmmCY&W6k?Li7Lg6QZ=aRi;GeRaja%y0FtJgB9hkUuJqD34+(vhRt*0?+sQzW-PK6hjTPtF;6c!<P}?f6Lb@Dj@P?n#FMk|SiuRHZk&M5Z6m37b_xsNfCl9`wGTjU&8ERyLkXLI67{9lZ{@wLv6P2<OdEMB+H2*aoVTM8walMaVmj)<fwpYYruZFaq?yS=vD9L=hpQR-8xT;cSx9?vnN*qCY990-))m92|G$Yc2hZP^U==uv`F+smJYy!N#1bl=l|Na$c4DDOi5=@kqH|QH>BYMjcF(gsc)a7P}b1hdk~Yi}n43Q7UauT(K<5z_scTVL+*0o8DW=oyUQM5;y=J4}3Uu1F>zB?-q7maD;`OT=Z@FHc8_ewrBH(NOh{oPo~O^cGnSP1wK?0wVHgWS;J6ingLK04{po`VerRz8kJ+q(SdnL%6bE^kdbhQ3ycin%#eH)z#U1wu0d?3D{$2<+y5XdAQL+jDorTK5SBLb!NcyC_UMtDN@$vsUE}})YP%^ijfW!I5w_LV(n?*4LT3QUc#AG7e09@h_C>2!BIGAB7B-*+TN1^=Ek>@LoO*%9ZU(L8g8PE?5p6UW0*y*W3k}K=fk`;rKufZ@-e|3n9N-veq9n<JKA4NqrC8;xeY>IEp|GB4fjouC(|T*slhI}Xp=C{63adRNAQPqd=yVvE^4LzJL1?-ubQy^gt(2xic7#IY&<g8X3XAE4Dr&c<8-=A4$Ae!v8>pdn2xts?z{HD~UdHu}(JBcD8jzWgT@52Rr~|(gCR$9L6QYc!tse}ste~c4`{+J2FVUZA#qWY5N1zQb3TcN;@nAIJmgaQSw++aamro|VF#={7+nb}{4B!F90f$VmAkBqcB1M}Ru-G-Ny2dz3&4GYQu{YA>QDnlUIYjY?T6hembu7G)rbiSf=tQg9jSdfYn$IFawIc+=jhvc264`+*7Dl1i=#@J<QWuL7CrdN3V3<jtt7N<0ywsg-N$@bGnb{x$OuHG{s<__j*E!45!fP#Yo@hp~+l;O==Mlxa7exisc3K7zgu+=ii>K*P-RwI`rAm^%6gEM_k$rMBMipA^`5*LJE^7xI?TcXHxDk3NA00wBri7*k^s1BwbS-%`H}mD~^poR11?s%o{w$U1wM?}{LdiCPpPMfom`S_6>8+$*><EaG6^794GU&JFEvU$V0)J{ExI-k8a&gyM@_)^!1oMvL-x+V9CwotPg}a2?NK!SP0pR3SUg}*des-0_S&prHdla3SPq41(Y{5BB&x&$>Oe>3n5*$M(w8EpXQdyo_bu%J*GgO-H^|-N_d1qW39G6*6C5`T;n-`mFsm1gpo$QC{H}YJkFG2_2imb}egzOzDA1*Ftl%nIQsr5}zi(6by#)FOeD1>(5bcX~=KCE$`_HgXP(vPrv(B74c!gj4p?G+c<fs_(n5V_XYm-Kz(3_Z&}QSz+L==!FX5$BEn$qp^e<yBcLAj{iDMvPwfSM?|!j<9UT^IJM;v%r<s<)fBzCyoivJ)I*J@*$UD<$O^7jf1mdiL;~a9pn;(0LN9eG>JdY1A(|jTeVrKTwe2W1JQ18KRbSi$l{lsJn(ia(AM;U`LJPeBG9sQ8<0kd|F*cgIbMq>jwcUq<AQF<en-XK%_u8XlIQDMms0gr<OXhT42F4TA|g?CQMBW-;<Za51a%;okWG1zu4+E=lRx&agZ29-X&r5Zrf5TIA|xv~<w-lYeM~6_j7m#aMda;mJl~dCa1NslCLTbGG&bpQD<mbbBjW)7B4%~dGr^@IrLOrCW-KL5Tvn6TWZyb7oP}^UYt_rF<pA6-Ji?SSq6-O5Fmqw;c%#RGc=s%p1F~d(s=;7VB}NxkWIbZDO`mK8V^5kJ@<(TWIm-FQa9K?iRFTCX1o!+hHjc2mxhLfIOCRvq)P)O~APu<Ht(FW;p}DXsEmxp**`dx$th3<oO2{@Cw;)Ae>PXdis;79><V=gz@3(Pa6yv~^C{T*~BkI+byN6so9nV5=oCn%#>ZBx~2mIg$fZa$Ue^s)v#OWwg6M{55uV22q%9C>zC@YnMTb)#fk`GO+0WXjmt}C1dk#Ql`rG0Qg!sVEFHzTF!+`U=W%58q6hzz_v-SU#!0BHA6%M9tpZU><7-daZoTH=_`%n-J`l5<k`9~Wa<3h{sHLWAW5c97!H0ye@JI~W7t4%0c|F_qabPrxpBTu;zimkDD86c`Ih2WMcowzLt`M;_m0za~G`?xdF?^*+_jmq4BhkxeZJ734q-7a*To&rUI7crfV^hlxP6;msw8I?-1HU|wHuvj-+vo+{~Xfjf|TZ+}OekY`2!Q7V+D=YLcax1)O&g{E8C2)-+I=O%l!y$>F~`?1+O+zno$K06(o>ew@CP*9Z+!)E%W*rLUH<s%=Qjb(M<yUn5`Mr2Z0Bqio^0D^*+r9_PdSXp8m3kjr%+2-kUZZH~n2_phXK#|lf!)|Eg8In56O(M46o2??|(l^!y!lL*f=&+eWUGawxb1B-1L-(vs)t^3ZbS}`^vEl}gG+%o_xEjCPfg=KdbOKPR<Bp!Agk6#Js?ucu>d7Qdv|c+V*-47K0z;;xN0_q6I15&kXk-?ml=Z&6v0AT48xcfNr&~#sEG)At6ZT6msubem0?GsiU=^L}^MJR1>@ty9almeCm@2+Udv*juUdp>tE^hQS(JnZ(u}rwRpXOp{-@dZnDE-YwoZCRp#(k#n&Jpf&$HE9*v@br2HEPHYv+;W#s87Tb(Nh*uGalIoGzK=wE{j1%o$W#*3#Zo`Vhq}&^bLVTAO$=q;vh0ZSbyKjX%lLZ5izi?AP`)@JQ}{oG)U+WapJ6+Ot7&ZIDrBtAsI{<w!uXSQ&BBnlRiJ5jq96&k40iHL}m7J<pW_2*YO%I99?uqz0!SbRq<oP8(=A{6gkW(2O}?jGoT7F(zIwcyPP)FVALdm(QN8$=*v@E9r{dRU`=|i05rWoXt@%uidq)8ixh3J=yk|^icx2?c{R8u4uW=Ao8%TSdOipjM}LqJ_6hBA3J6HmF1->nLO(B(^H;O~O^8d!^}ls(Tdo@B$C|9KPjxZvFhtcziRV>(w$(3nXZheTjK;JqI_Z^?8<r{Ba&DeP*b*Cx)UDnz2dGVICuts6)ptt$s>UU*Z=A+}HaPhFi+|P4vsss3-?fbd^p>q3%hB<lhLOL2h%bE1uZ-lah4H4j6nru8Jd^*PvpZdsP{A=lEJNil#~RKT;7i%<RO#(1xEmJuLusKj{Y<0O>bHC`_7KE4+g+44*X_Gk0`!6?OrsQiZF43A;6!(+2(mCD2BAO#jy0!dAeMxoSDV{@rFRT_;<h`eLEnxu-wGgYoK@`ZAiE5z%@ED4+FI0i7TV<+w5)HRni?&_u!?S0DLF${o79LJdVyK<92muZ1ea-5CPa2jj@>Jm2ep$GW`E;R#b~Ae{%pN1lCSbQLN0oo*R>=+dWXy-o6U%<pw7Wl63?D0B9i1AJ@X=B47&i0X~=Ot4IqhjjX?2{YI8~J5KWsM17}sJ%Ot=|lfPaBk8KI&9Yyt~!^UnxNL*L3bj|?d&o0(S!Vc&6NC{h;>oUQLOIZbH(k?f0!@JVBd#+W+C^a6U6;MLD2O$_epUF<!==MD_p4vq^L%u-~>umj@8Va=zN+)HZCH?vA=H@Kwe)?IOJqu`z@D>9}(S=G{)pe~sv3rQ3xGMjm&Njw9m6EEaVtOI|B)y8$_^Qadlad!-d?w1dG`S;6J&v-3_L5X8<b2f2l0$L2S;F*2({JIQZ)GovkP;zyQa$8U?H<j)i1;YVN|8*%8ewdrmroLFC&b96ir-JXlPVUf_dyn+hEJFzu#ya~xb>Cl|I65fHQ^`ZuZ(Zsgfd3XE`C?&W7UCS?W~n_X*&;43M)j&L6Si+R+Rt{E{sYe<!zCXrPwXF8BOEuwwpUgWa5e!r$p#=p^Ms<5Dzm{jVLoX4%#pVT{y7~P`a!jG}zW9hj-Ao8-_gLaz*ayP1q#L$wLv!lf|)%=n2jrGxNKm09Y<OjwydAQfet9ZO`u#)4p++oYQQZ^D>-AgFOwTbJ_F}ZZRqI12z*o;#obOqgX=2<#qkj_7hTY6OS8pi;TC(?dmhMEXUGLN+l$58wrcJ#c*SAwVHe^TTn^f7^(<vg$lfuf^)4VZtQ)UTp;UD1Z2BZ!P?2JW({bmZ(<o|X!kJ9W%V`Nt61bvUM@8+V;!WFFq0lCM3Na_i`Ck)tZ-h1(?FA$;$1Lge}>wX`VEKrm_<X$7?B7p!ZXZSGjWm^b>+nU+;US?=*Xn1<Q|W!S{BphW3ppIaH1wEl6sNC*t6k+t{osCyYlQdRTEQNhFn@)f*%=7=m<Szs!dFOYRt6}nsRzYjLi}8q9#cGHr?}z8`eCd+d8{=dEv-Ash;YpXc(A3g$rV@vDVV-(!wORTcMros!jpP{0^Aa0u=`oOJ?@zb7>v9>NIFNo7^+Bx$*-&YYMSkBt&W_-i@5Lqz-Rt-j`+l6#K_=ZOv3{%XWUU$oX|w=0nMX$`z4bL#1a^v~p9Y!r_4DBvI3hfuNugA+cS+O*iy;`}A90aPhRG8*3Bnu}BZ&qt4m@lRMd|y&n@TLUMW|qJoe_9#Kxwu|{~y9ssGmw%iX%Y8-`4)i0huf)bbajgDu+aa-n?$V4C86~=%@Z!3xFYcVww9I3F1Sd!^(@8l^up26nMUCNV??I+XmD@?XL{75u-63jf%SJ$@*V+P`g8J4cYbidVW5`v{GW2+>{iNf1T(V6%;1`il9%u|?atlymzOeKG)A*x}@b%0|&LsAlB9L<7d=CakvX<|>Ny$MnGtjf9%xMT)Fc-BdULAcuD(ySbyz5M}n$WlQ9PA_7Iv#3IFJCK^XpyrIjnMU{WLZs0!2V%vu4J;L*<?b2&e>^5l3d?C~R<+ofyh_3-p(1<<7jK_&?rjD)jzcsOjj1*!1N%md1p^`&aKZUh#T4j09z9E{7%h8vpvl*Iexr3zi%9L^wK%z6*zbvyw|a)8JQ3J<1d3h}yhzbDU=TC24I1V3m@+MNb@||Bcm*5H+rTr1JSi8?^r9Q(|ERaWI+VohQ}d_Xm648UXcW^k&Tp7qCVAsY-cADUu}yJ`Ot`fh{cg;ZIjXO$0)IJd+axO!ltWC)g$xJmV?9JU#EkPch7O3CnW91}7GD*pjW?*odH0cu@0?77J~RN~$gD)Rbk`-sp<U*7GDM+f%CG6DCRyNBM$${7u(1DqoG*cNIi>R>u%Rq99?z4$!RQ?MfjX?@^sUl!Via1J>7j*04jTPaYnL>|0S^R+q(6Yep40F$IPVl1OyI)wWFRXPwc9$3@f>l;D+a}C$i^+in5fwwt9S`)ZOu_rrp79EHAJ94hEa*z<z8;uqK$MmMf9RCngNQZ41kp)^!xeKBDmAAN7ojIEa!1+78(>;4c~71jR=dZ-OH?L_<juul&_{EAnc3P{ff)I9Ewq{zyY%QU^#MtFF3%vVChs(Li2)E_+r-|3+>2-@GKFchl&vJordEl7akyv1t!mou(d1!9EPs~CrJ@QR5UC)fl}H}PWyAuD;<m69dQi)j-GE|#}s<FCDB~P+c9bBAetEP#tJE6ssTf#a8}6F0f0RrHH6U@a!xU1iPC)rpy<TWFQ-Nl06S$^D&=JIfEv?QgraotNQmeR+0n7r^c%{%Fj-S@GPtf}pvS&&cjXJ(aeBxKp|nGUvi$W`+jKapHBVRm0HwW87D_02Kvrrb{u&a<{QYiyyC1)Kku!DnQl~(kq%2BwFV1PoJr*+k4XsYXF|U;l7v0Ud#I>irn7r+5XE0{Q)ViiCAwvh6++c?Gs$9{_zAEKjGMJ!Po!DX-RQuKyMOLYAG*~>*n2O?HQge#%yo>I=g~6Ilo(klGXhp8&mF7+1C-oDJ?nQn>wMZ5`I;F``uhJzBlwDJ(>J0&8d5?kM!XaymoYL3f$a|FV7snu87ys8ZzGl6Z`<)`vqOuOVS_xILr<5ZlPzlF2u$Bb?WlB_z25%aHmCHU-*Vr~!lOQTmP)W40>LbAb1(IDyLuSbNt0@BNP`*Y<bllq#qW($5t-7d)l!>1$T&%w{Mi@{b7FvD&=~BhkuR8P0%TqQL)}bI#>@-3jhycp7!6{(Q4xIwSP{X29!Sbk6tOh*}p~i}&ETM9N*x#Y>d|wgX?gwfYk}^JNqDqOs<`d9UCc3IzD3e?x?>_9=<AiREh?>VbjA*)7sYTPg%F|+%gYUdWrO++JUCp*7oJ8Bd*&-#@RZA~AKukh(C*|wb-M7_3i#z{vQT&TO!NkR))a9GfYul$xgyA=TGPyylh@)8SyFXiE3ZU5OD>)(Y;%u}z%bv6fpRwMaDobA(1wRo(im9Bcg%vp3ai-GF*bH1_@@;iG8=z^@!PCV{GB|MS3G2A*$2VKpWPUp)eHiy|BT$Kt<qBb6_~7WXI9W5fJ}~<OFwVtu^|M_(qW=lAcBaQ&>}K0T=z?T96B<iFr^oE1PBrOLO?6PSc%*h|r+Hm>o;hWYbay{~x(G@yaiJ5;Y>citAn#@VX{C<49%K`0STQHodnTQ8d8&qIor*<z5Gc`mScfHPZ{|j603MJGO66|XwPA|GyOSlkFYrM0ao|m94M?YPM7?C1qMS`O$SadH8v8e?WQCj`RNMTfB}kp7O{SRtnr4QQbE0AOmfPkOx(a~S7GvL}yHybtw88}Bz$%(0h*k=Zp`}If#EQ5nQPtzRa(R{>_t7=KIb^SC-@4WGE2VTq;f?KJN);DiKndZDx)n^=Co5NTGHT8W?AY^bVIwiCnLSVWk~*&lOU48V4c!6TPD81P?}(;?&DBYJA)D5Gb_(J|dBOGrC#8wSx9t<4HdOCIcdg}DlY$`%pp$sNC0K&mHNP!&>Kf8_mQ8Jr)4-P3cHyb8slh#RIyDV@wRqJNRc{uLnZ^7yYGXH!<*Zqlg{-<wadc3Z)2{Yx@cVZ)_aGNcDJg2vPP@cLfZe2?K7VGurW+*6QcXfvo;BI+A<Y_c9p#=jC{!4|-Mfc{iBykYRH=?jj8gkwlGwp>Ucf*as6LTmqlBE^C7{-Hd(t_*xB7EJ4pQ%aQJ*sTGz*rAw4-Fi?&13C+Mw}~8Pc2-!KMLzcNA~qSF4bY%<wXIaO_ybEQzHYZ4u-adYE;zfuH_+QOlasY>t=^<8J!Ba+BbV=BT8(0(=6Tz+l02`sWR($uX=cqUBy8dY{~?rqh<-KpIjX?TOQ@8hQXB*+BYW4<+b^J#PA#JeS<J^M1Bg4Sn;-@iSrp-_wtl6^x|~iig8RPtv)V=)CSlAzL^SQIw0RPncXD-T?+P<+a1*bwWOlsik40$#wAz5Q}=N;3IRKisbIR$V3Z#F-&A}u~LpzRvn8Aqxu|sHC`eiu3F-SnQO6(8@CG?`j^f^vXZS(vkpR_q&S+pkb>3X{oqtpAgILcnEq9%m}FW^$fjILft9KK@QOOozGr!LZ~0ehKW95!s$F<<NT4nLG^I*4p`ocYxHN60FtU1Ec28j&*?soyw^r}U;SU~a`1h}9>dnUT9zOXfQA028o8PkG-M-YnzK4${KbSLD7~0*ZnB$rxx6f)^?GF>M>RF8o%;2*aSH1_k+!!g7aNA_H+a`fM%5xj*yjcPpIreNS=_0&|wc@&&uo}qJ-c)v>t#s~FNk=71-!EOa)bq_cYAQ~fT&+)R&@AGQi8E;uZn2z?;kueB7zu%U_F2LPpy_YUYGtAS=ij{eTG-Xbq!UlE3eimp8Nf5Jye{OMVVniNyM2u4b@k#S|HJr7uyo)XkdeW05JtWfN&fA^3uQQ`{+;YX!Nk$p+e7@c>$uW(ZKWJE{m#4NV#t*?uwsEMx)e^AHgHs?32X>LeoViT+tBMEV}9DV-+ega+f)DW!vo2_`PJ<Y;^MSryXkU)MyAEq2=v2K=$zi==}9&J%Hy}b{ok9{fB)sDmuH`zU9VQx*Q@RNU;gvW>sN2D;vX+x{^f_)uYUfovrqqdw%%;secHQE`E>Q}^VXM_hZh!$cdu;L+rulD?_aol_?yMq`O$l?uZ}-%bA5eyVSRY-#k*f^-~My|tLxhr*7Fx0{`lhX$M1i$T+H6OT%ot#{jhoemNj_GYVwxfemg(=Y5U2Wt54qi`pb|1ST8^M+smI{zgjHcP1KJ+{rcC}uink{bQS*a{TE+;{=?g~`0o2-Ot)GNyDL<F_SF}E{_K6$^kP40&pf>R?VonLuP47aE^J=TUjFm$%NJiw-4dEQ(l#uG-hI;opGFk=N(;&>hu8iv5PS^HmqIx6!qvmya2U$_MOnC|qysTWoYDLDKYUx^5&=Hg3<I(uW(8jRYNed5C0E{V)s#*GH}TSt$9ec3Cg_+aXkeZa<>uHB(^|%TtZCggY&~3Usb~i#?ExukZ;Fs*V-@Hm{!+d`ZLtl-&CRX0_uI5@DicZgJ&J5&ttgeUQq8{Iun7>{Wx_2r`(rwPMAHA^O?d>W;Vg06MlrCC_rv6sa!AN;TmW_#z9IaS-jT~_`0wXqTa&{ex~C_naXgIq)cFbP93GE80#^ITqK7df;7vAzCa?S@#uh1m$BEVeiIl9eGM8qBw7D_DzGuDvjrE;h<GP;S=D{B09hMkXgr$-918M06vc1n3c@8W_Noxg}uihxx09{>nRQiEzE@BwIO2Si#|K&84Rt_pEcS@Y!c?0!vRT(==sDrZoBp^>l^+R0zJb_@9qx|M?@spgmGV0^fP?Z2@j~rNk{Iv0(CO`L~6e2{h7hOStL)bAMY<-V7284Jbn|=DG>4$d0*5uJJI!3Ta-NnZv!`Dv|J)vF(`9Y=0qsH2ey@)H}5Er9~uIV6Qh0^X=3@{j<rxY0!iJd*C{`_h_f^i=|_ILKNIvy2AF8yL{+o&X;L|Gdf59vHGMPTGqt%>m!de1CbZ@v#3kQ!wB2*tfAD0`qgt-nL2hct6~F*S8%@W|8}YQoygi!mT5{<CfbXHvKsVgVKaWluniD7J6n7H#Mskp}seRtuw?Oh_*&J(!h=@x=`eyjlv-$0*_B$M!-z8z)c#jMyt3^$8=XyLjoieXLSAa`dLJ$hQlwiiDZuTW}1hE-~fAfuU5tlZ_s%&C#pwqJt5Rc%-yJP+AC3m0_Z|MQ$8CR52>RM7;OU<jH-A*{D^|szLjJxqepm0dxvRGjThVYP5)~S>KPPQr3*=X0tKJT{cy!S+Gm=au|88j^V|9EHyoS;c9C~HIEdlXxImIpIqJaJyVK_n@&8bJXsP|mAPqp=4}3M+TNZJpPv~nNM6OR%pk&+qMMsicV<_y2RPC9Imz8GPS^I#ufz^@PEu24?jx0A*bttP0`?NO!%6@cJ&D1~-Stz8jzh|Mp_z;c{4gyla{}0KiJC=~k94^17EP)UlOzoe>l52>xX}7WC{Lw~8TIqDL1R16*(;Am(&`F_wv|p)yLtvu7tKA(Zxz+-0!u6-8ZpTotSIzBe_J5SqNdM>L%byjdEpT10qm$Ip}8UUu!P5>Th6R*45^B28m8f%9yxb~b7{nax?6J}oN)y&&WI8cp}gTz1AbKg-)NjLZZGDFw>2E8btQ3Jk4y;{7Fto>w|qzt1dd99a;J!%37!IV^W#S{NhwJ51krXo*F*qPk002se)wFZ6ENiK+qp;dyPY8jm3}3OzuIE>slO4Zg+JOw7DOp~FLLZxf(YpFw@CkD=$8{s-pTdS8|OrzYU?|9l5+CG&NPOYq-rKBU^gO%3m|HM-bVZ+&PS+<#-;^Fwv&22wM2+RwP-}z$-E?K?3KmGl{kzAB)Y1ZpE;(_vWLgW-ElP0CCg0lQd*z>nTkOA)XRq7gDaNTFKoBTgVzckd+UNol0=&)I;bIdzY`q2lWvhbVH0R_M=@iPw6z!(*n48Ljzd;%F-v;<Au=TsxhYYOiRp7%kmh)>q->O%7in9QbZqXq6NH$z^2VM55w)GX2cjm}1JoIW?$w$JG!apX(z9q2xU;E?HPI>n6X#|k)8I{A3*9;aWfN)rVDt@w=>lhr%+*mwXHlM+xtBZ`X=$g~611!Bc;Zk@CZ!Ouv9{Hi%90;GrcL#A3eu?)XAn}e6&{o35ypM{X@bG#iSP!g;}Qy?1swt9fk_GJFt?X+W9)avY%=_eFkY)88b!eeH3N9lu=kDb9n{?qLCBgV{e<(8Bss2<0yu**0Dx(gi{V;F5Nx=`?7iH-5_g&`dhqaXBc~KPD6~sf$odCxRnTiY=&$pz3zouJbt{7n^3*(Yd9Kp3n%dFiGmDJGBrg{d3wp8nIoobA1&y(EGHQjwc8gFl1fMi?evQ%F;%*Cj#dMG+W&<b@O6sP}OJ+$P-l0W=ngzz>X{j|e7kHG!*Mxej#_}V~{jamUjY1_7gGynkY;V1EC>}@(T8N01B&1~F)02hCrP29Ob~-zyt`x+Zx~{75=}O35Dc7iB5n9A!j!L~@3zXJJn8fEfDn>AV!eQyT|0(nK=TwxZwFWsqH*&3*nrzq-I)!>xnA-KUzIs@PFk~_RE=wDA!5$S_22F(Y=>5teG_$~%M)bI3ScN#q=19H>Zy6_ls%-&~8l(w2Es~3WU2K1M?k7Cs_*JrxYs*biq`mHRED1qEKz9K=U>61=O7l39Y9{8v`snr6K%TmjgMmH;nagfA(c2s#M^)?u$mN7_aK(@pKrRB)5`|sUa>1s4ebN#@>*csM3_aEE>0Z+>udWz)*pS!)n>e*o5qgK86U{Gg3!<y%%8rYhPOAdVy*t?(43LtS_B!6iYw_Yqw{!YvBWsCd?kFcf*yu-l;NqsFPEENCjPl3WP4dPKdL@@efiZCpoNl>obe*yCV5T`V-<jL!%vqGTA7JaT)xdXNxEmEt7WW!TE=U3GP&<skz7SDlbvLfW&1kVm#3BMb9CIYKbV#b#DX~xnBb}$`bxbJb)R6dAe`Qcl_;OVRj5vy<gGaNyad|4fM{Z|)?#KY1ocZza9Els%@u})TCX6@Il@S9lM@W*X>TY(4Ob?(FIjiDO!5i8==zT#ON_dy7Y&@4f1a?w7dZQ=(;>J+?=6z8_vN+=Q1}c(-QR|f><Q+#_py@Abjw6Kj0rb3C=0I6N5wD_FW9R)Vaat7i4}eB+)^}1Z%C^dE@X&WxzSg48XmOe}080mWXHj7m8FI|&F;4f$PTDcn{S+)eZ+oI`O%lDPqAN=~mf+fwf&sf2!B;%)9-~<^8YQucC$Iuun@RzR5Si4kP4BJb&f{1@2_FED2R<CSf!MamcMCf&;t$MDF8Vfon-~pydEaAu3aC>}elq!4+FeJ$m9&`?+#mT+v-YXdGy}jP9^9A>!tjsrOe)7_qXYBk*RL1}WE@_cj^`@DJqEF2NIDA843@6tl;&EBh<F-~XoZYKm@J2+3#CSbA<?YWV%i2Q?XsGk3A<qu%Ne**OAu)mZU8_T?$g2D5nXJB7JRXrfuf(mLAW;h1={mK#BoZbGu9knV=yN)lTgGT9R5>ZABwa_67<BH$8o*S)rxsIn_89u+xR)%iq_SIjHA5?s3a!M3Mp>YC9y&4Y~YBa16BkzC~Q+?b?F$0(9m01#_2Fw)i`CYQKj#oD(N72tmWOop1f=q-|DL}t)>S`z#ft`PwN{#A;$O#Z>RE5epPG7k%AruKJp{eU)DDjdvFolRW7A4MDez?aU4m?R~#d-UW#Q_A!eZG`IX`IXi*%q7vFa`AJPAsRDUX1@ibXJXs=iK3R7LalV<v@Jnqr65aASSjT9|3y%aG3y`e-WTy${}II}$+j-yy<VFJnq$b=_~;DFduGQ^Wo*)u5b2ZykV8Wf$vZ3%l#{nISGF@8Y0oLY@m;B3|~pEiGDiDLzeI)T!BE{u^frH!s9mM+rLenBTv?`~2sB~y50*TL>!6+2s=)JnLEM~BL|A%%MlHPi((!Me5IC8Z4A(toXM1_gTkLUEprm76&sCy{NbzUqC6nS<aNCTgP60d%2`9z*p`*+<V3jtmqeT1=Y-HA_YH(_sTrbXLf4de%|$1c{X%nJ9`ZyxY0Wy!v=(8JTrrxQX;_zPwm83G?;uvIagR52k@xPwRPSd3#VtxszxbatH;TH%32@)f~qeJhbanOp$B7^(+Piv%yMIh9-#?ys*TP##^-P*p&}9E!eZ>&@6K1H(82GE^i8~aidix(%54=I;pr)Ha%F*ELlzmL^8k0UTqTm!`F<=UEo1mDJu7+D~6P51TW@%1+i=lemb6dv|&upj%)$A5S*&Pg*6Oo0BO=g6vwYclQLH#lLHd#<3+0Yq3jj8ol#O7mgzZxGZKjqZ7{2f%!<SnPi|3dG|Y~OYhA<NwmFl&Y^$4^GYL90q&jjsqHD2ke+stnhf-Q3Y^@<<FLH>pv!0e6S7H`8CK2NrNeU{RpU5hoIM>S}r;k%gJTcF>O|69_lb`p+nxgiE@~Fpc9je;Boqa%o&a!lf$>H)%1CB<N{T5@|MB$qztX5jMTaBY5sB2?Tn$g1uoMP2l;x3blD`*@sUNj=h#?@1?BF$A23s>NgsfjY$@vbyX79`=VnU;}3!qzC)tV3Or#Vbs`l#jbQ0G@q7;lLufJS&k&{&Lg##jH{_lwUYHP&oA_zL~L10BS9^`Gi}GI~rq=_C{f}seIU*vLYAS_e%RX9yzb$s*7-G$ZpY=bms@+@K(47F0Gu=oqI-?>YI2?9*iaDg6c%Fp;)I#eF4ZG>R$*Zbbw>4!xMYVUSe(A(mJqSil{$RIZj2rxt+k*VpGfc<&2Uux2P#!)S@J!hpc!_Qpendi?x%JE5a6Gr$Lwill_{?XWso{Dy(<eqEV|4i|GDL$m}aQHG3THI+bZ3Nl&++q6$!P8eV!qOR-1ti>;}S5u{mZcR7u+VqwvfO_Xf&Rb&V-7$U{I_8sjHEo}e{3RhS_PSf4F066UIX|3}H1cgSKB>+BNQ|5Cj^=#inP*0ou7;YTt=BfG;a`HZl<qg^ZcM6DOBG%#0pqao;wA^VxPwDZE`v~flI0vM2$F1uS7X>yO!I7QbP%KRnr3zTDxk312b$+pszITmbN@$Rl#gk$>yQ0nsVz&ufod_Sar;F*Zx&6(vLXxWuxGcg;b=AGcP$rw4*C5~9KC66>>#(C^`=m^Rwjn7^`>w>&{%EGbQNepyKXZ<f_GXTIjdGO1;gE7Lgfb2)C^J!)NmR5QovRICBHWLxu&MUyj*`2HIeV*~9#Tzpdv7P;vSh(Egq(*O7ux-Z&}q|xaVX}J?H0z0QuWowC__pe)r>z7P*>3A4yqHKnUBoqkwogu7Ii7P7CAc!gd|tb4VE@I<EKyrTv<&-E?sqvC&s*cMU_H4!MvVj>;M`eGc<#fcN5aOvjIrb&X^m=%O8azR57_gPBFMCbS7C2Cn_}%;-*N$XWc18Q4V4OQrEg9ff!8ud2en{saHYE5xYU^<>8ZD3pV78jRKqAX|P|@?3WBN3dcwrH}X%3hj|p_*R0euiFwxIjqYvIv3@)$(NqYxa(^j)zRF1#{=I9=9GPH?CspJltV6hCmZQq@%La|v258Lsxk4|sN_!9X0#t_q8=!$+-0r%5<RvMGazkcC!RxVw<7@pLEPg*t*GB=y8mKH^!<VG8za!gcDAezJM8I`@2SidaxG4n5b0h|?jpChr!^Bbm9JWZIKSNvCs3wE6G~4O$U|o8PrCA5)&eXEb0bVo;EwS)4?{!=|eH#=Ip>M-SE7ddLQ+}@AG$z1fOQQu$eJOFi8}nso%4{G(;<Y3g6>(f4x!Ykcoy!iQU|AAcO@dNpD(J3xd(z*maIcvLAQ~=fW`QW^PQVDJA+WeMoTEV;4FJE!se0k%Vy8h<7oM)#Mp!jM{zu)bbkUfH`(&hSZLMofPLKGEyb|^<t^aZ!m*_ETLyN26+Do)2ekW4jo&X8kTRa!YSSY(Mn=S;YE~~%+5folZho<Zip$HA~ww?UcT<iBxD~`x5Docid$c;dim=O*`atYKCp|>$bxtQ?$TN-zc4S!9Mt>v0~Doh@FPXd3n8IbCzaCDVVEq<bKlC<dbM3khWK-uYON2X$k-X&r%vy}}n9mKev792f3p0R?CCGiJadZE1MwF)Is1j!Kpv-MB|9H7YAF#Jst;F^|}3(}an1z4ME=_-t+tCnu`^g@&#9*R<kW&IeK=_MtlWp6=9Rn%$R_jm~=+@nd5EIdo0R-#zF@8WuZuY#nb@H#07XKh_D*&I*aC=?|QobFQ1C!Q~soCIO5i#Eh-N+i;l1)(hfF0m9&Gk^${8rWbrIoX7?EhQZ~%I-${g5}=P#?YF1c7m1y2|XEqSGpxHZ@M~vGz%K7i%A7II`AlOP3XoaYe8JoYZq6t&%|0N6A0ic(;_LaR%?Z5qxzQ0lP7G`ea~zp;GQBk)ZS8Ci8)`YQt4Wpf~Pb^HEx5nSro7dJ>Q=xg~{>x7~4IDwUdSDGQ>zu3f1&`Nji=5J3Ak5kx38ovG`4f{Y6~s8P<Z9(l=UoxSKpRrvGf+)qe)g(0manCC3otq0VYmOu=tVKL-9d<wM_O?1^m92AJf)IJhv1{5#tHRfWhtU-8@vgH?M$tW(R>$v$w!G-N12R(jNtvG#nR1dqZ*n~;dXsA^FjN|1^SQ}TCcsLGI3hFnw47>1^wV9|p`(pJR8eo75UM}PPE7ymkX1sE0tZf8i3j62@AfR$_lQtjksFsRo-rOAkjq=2iDj2cQ(#W!&7r9@A-^iWd;IobV#BCmE4?r#XR7;52fEP$F7Mzhu{D1_R!V}Y)xD;=|J{w%n`s(TU?U!5B$-IVb|+ZO5}H<a4<9zB8h2255;nD?ZGht_$tESEC#@E#QQ+emWI5x}G<7)|4@55_0ynVBL@(95Ko?x|goq}tgtFh<(f9fwZri&Q5xCA}xvCv-N)F|~^f3oADLalEE8bFEq-MKlAuO-;km2C;gzmb+HZNM#CXaJt;YnnwJWLMF!g@zOR^CD93h7stitPysuy=g2J8<rr+R#WUo>5z@IXrOa{<UREEV*q7OQbeU-ubalRJE>W~`5>H0Ot#q+1VNHba$8jVg;l^EY6|<%;)C}}Y@wR9#UlP*NYA0UD##4KratE#C7WEZ?u4;!n+jKOGqhMBq{ExW|LdMEe>)8#3Qpww}gNG@NL<?K9qYn=SU25!<R9xsDsA2_!mZ;N$h9MJaOkbGg$9)g!yxjudy!b9tK1`CvHKpcow6VNAJEfpwN=n=4=-u?SNx<!%5vd7GMm{OMm1ccowDfv5SAxm8Kv5e)hvF*U3J>jg!?kcV>jLe$kQ2#L-QsCShh)Y8i(02o3BIiny0+%v+j@XlRoD@^Fw!FZV&IF3(u9KDDYKwgM92?GD5e1ZHKi!fMpj|A$@<VI7fzOih`7K!nDCT0Rj{xkTf*t8%y>sC(W};4)BZp`h&YQuZ0sm&@37|CmK1}TPx7udRDiV-Fw-mNBDMj8$B>y*c5jB$0pfk4Q%pO86z;tPE;ityuKLAwFTx<&G`Z^L3MUw->D<1v+GeTB3s&UO=bx*MXnRMgbjg{6+#^Y~Uh;#3q#MZ)BpOysnZ;KFiA(uLFx0_wLwLFaBs^#?ietA^CF550?sGow0Z-|{!GkSzz%%sypH!Ub6Tjh@P4E;&NKltyp(oM(jLvYjP*^>2&{1w072dzj`EvIl%JRmc;A_J1^voh%f3`{by3Pw^=u6TrEeee>s@?tX<a&)$9JmF7Yog);=Odv&Wqdt`hRV}1R$^`E*{ASig&Cu(*SFE{U)6%3Pf==%Nn-<mo{J|owL=Kp>7}LquXnp+sPJ>JyAx%yPa!ZBTi&e)ySqP=j>Dqko$C7iM4opX5=xqN7`~W?xZ9%LU9%Ngu`EMNWvI5G7?bT7cxLG-2jg}3Xnwa3JH;js<V*!cyd(=Z=o{B1!VLov3Ib=AYGFUo>*{8V)tT0qvdot-aj+JbGV;bzL|8$dNa$g_DW%@2n=W%yUOew@b~#)uct+d(ISCAu5#qSM_*HE+nNn58jAblF!e9`<sV*ih;~~OZGwi^kjt~i(u!{<qq<Q3lqUvmCBs5hHuXbx*iFYP^1d2;di(;^qH6Yv7PvNj7%O%0H$TV-1<em;%sl-ut?S^RdZAx%PO8pv$kAtv#uXvicP8(`>XBuZeY!^P{r1FB;nAZ_xe)MOdf2vyAR|Jc+yI$e_v_b=lwnxpz<uPEsWk6H=U~CefL?=EoEgJ2h*k_zr^chr=wbK2m_KCK0b!tz=&PGvPO6(D>6h|q3i;}YXTB+cM^+YRlg^GSxNio;aCtuwdirOG_5q5c|XC|@-^U%m~2cmlNxG98^zs3G7@flp}Yh)oNDe%W<9C=$2gIwO|XBk(ux2{pW@oHK?`qt8<^<q@6eXr`qt0Qf$Pnv=&tKM2dmy1q8E8E>M{8*D)mPCuqHkcTDTC5F#m>Rih^KTl!+6}V|Mq)`fwGot!FB2g^3U;n6rp%(Gug8klVQSl8#5;lNn?IUjx5ds4$!aa08fX!tzi^^C$s1i~>}xWb?qdmL!xBCoHDT@)D+y9H8gf*(&)Q?rML}p!(xtcC%vU2gbApl`D-)zr93fOT*5h(fd+2gBON5b=2e2WY@@Uvs>&MV4r9BX6d+?=p&L7)HaXonaOfOC5j3nUer3P<^Bj%j9*v~F~oES}0QiEcVzjh)HSfTB!-XZF}%pdMLT6uL-X^fTWY7(<rhr^65#d$Wgq8}Qe?_`fVsY9?Vqo+987*NV;RA=B`)v*HSj1su%^f^j!lbp@CsOeJoO;50pMYq;2wK(8=jru@ki%6afq6O5UU&+8m+Wm)+W9B-b%tf>nD>BD&?jrXl!tGXK9Z*pzaGyvRPM<>q?MQ<Lu=i6gHXcJt`n*-A^5{gRVA4&N1&v9UBx=?lpa7UD5Fy4#(nF@{L{Ssif|OGEJf#~jZs|%&HtkW%l+9AAo=3S#7Joft@gs_BLaO0#VuKna58bFDWVMzpg)PDR!y9cky5T({j?k^Gt64SCKA|YF?J1D&86f0eqI_UAl4)FyGkbcRaMf1blu^cUZku&E3nqFLh0Ev230#g<{J-SlW79{&4BNW;0Gm92@7|NAx13DHRBX@5F;EMm%-ZIc6_?3@-JaPfMG-N9Vl~bZP!?B@BuX?txLuD&q}^#LnVxldi&kAfnaBdn^kiO50U}{|Rzb2epHR{R5OiK$v8~5i@N}Z4)`xXvJVn^Nx@Gya_oNRlV?%ZBc##)h?ZaeD{=L*NB9S_*t;<?Movsw_mQ_>@<Xl5RpGbkYJF5Y~S);nx#mjAhZ%XNqtBd6bR>|2~BbEp(VC|V>Xo(34XNU^4cg^vrHTY2W%uEJO1f$f{Kb>?+vhcQU8M=`)-erzb#<EqbTIvPSJ0b6k1WyCD%NqDVeC#gMy8^*wD)UC`h77CVr)1eImXuSIG{LPtC=bHuntmtVg#7Xfya~;!P%wwm+qfjwO<L8})mRcqQdeVET7F{=@AIkK8*?>6AL{<23Xv43Ve0K<dlP7R4ApSiU@s{V@O;(sWtLzI-V$fvZ4$7O@*fnqLRVZ*ZV&1VUP)Lp-3q|vpc*nXz#^HfRevBYVxkzTsEDJIWpWjfJkSLPO{Olv4T53<oi*9;s80&QnhCmkA+=2+35@$X9#kov3PjWt6hwxNBsF+0>d<(tGPcM#&r11kS<7wa2k7r1fbG-gV*?(bR4%5JFc7THAyIs!6fhwzJZD)r3i9Jt4eyE2-X-@-(2j~U1Tk@J$-q$5^xf9iO1Ym{c>xM+V)#MPAgc)AjnG1;vUMDqP?X^d*RgicEQz_3pvdi+c)YVmtOc)b<~9|4oDIfWpZ&qW(6(v6I_g=w%_FZ36Vk<17uW*s`e)*qS3reQ+7fGR(kt^Ya%DgdH3rc=isbU)+M$G5Oqn(~O1U$#PERFg=f`jgpJ<A5dScE8mYKYhCyB7=!UJ9Py>pO*r07?z=0R9jy_F@W6$2w}fw{e0-DI?QYm*6{fK|iHv~~@a$`=>Uh6a9bNt4O;UWxA*?O*8aV&Qb#R6!tDtFQ*hsr-ee$YfX4O43YUbiXqNF(mZaTV<lW3H)++wi?zfW`Ewh4-cO=$;$3uVldX>P3;f0#f8C?G&=CR^aqa$ar?JuTwp?g)b^tftNj5MK%c?55KZD442x~N=P)cFu+%%gNLeYRg;6b&b!7W~QDa6lqdc_wEGpS#qQ)xYy!|>NngLxN<n|c18rm8md$Y!I7BgeX7sHVG5leSW?W7(pXM{<<{AsLiZ0#gki}_;S#Z%U&AuAy06>}$D-CnttCh=J2foj)SCzg}~b0n_~0@(9yAIF`wul@X+7hgwD{ro{I%10b4Yvj0>62PV}_RNa%16Z*1E4^!Ne~23H$yeSjaQ{^*wbV9Ph|tb9v|`a*ds`M}qWG1Rr<k+l+HWOi3~^i=e4VsmdeoJ+hs)s={ReNO9ckwsYs-xZ?wtTW`Nre7zWv{u*MI-zr<Z4+o?Wk2tLyFJ@?ZY*&FfchFXSIDU;gEX*ROv5ud`48dA7dZy!*6wpYrMA;@t<XudffUtlzz`S-yYE-B-VRVLktw&HD1;mD?8{KIHQ3{OG-x>%*_M?_St0FAguPZeLg)f8lnyI{x_WYBl?e_44q-_3aC*$!DBT|8~8dzx6VD>vnTI72EB@bza=Rb$NF4+i&M*KV5zDX8XyTUw`@WAM52Oe|!1!>sO2AyD9$hr(gg2`qjHNny%j;zW?IO&wqHkir;;I4E$EhXLrTS&%XNN&!4@|mtO2A+S-Sgzx~s0_x0o#$HmWiP4v&ZFJF8$mC|Uc5Zka^diPBW*(YeQco`T7PB!LCAsmb0>IJ<z_d|KVD2o(mZg~v1BoT2&@8AFMZG}sO^pM#wAhTmu8n&-ia>;7xfVW#UZNJe<X(+a8z6bcqvNB#mD_Eo{R9tnmmXT0LTE@=9)t1s=Fli4NL3>k#bR(;dClQ7#{hL_QYIdv9Qw+<9+*Br#FsPzmdD}$uZ`K!zRdt%KG;o&*x76&9>HHBF+lM#hfhyu=vDWK3*iwzX!~HOMr5sH18#gQtS>GU)m?|SbMRfhlKX&#gS%rB4JvoizVa%t_Pgv*hc=QplYDgA6j2QuMu^BXZ<u5U|NclTXv<67rY?T$h8cPQFY~S1%Vc)af|Hk?Pc4#%dt!O?w;W4o~6I&fC=>?M7){O)!vM73AW4ErBP|fTQM2WjquUx!O2Tev;XLM51{GiToCC=}>ftqBtmamq^NB5I}Je9a#4h{ek2v$m{$Ta1&oKYW_hN=WOd*s0S<EM@PH2JyD<fxAk>_t~l;81vs2V37GjsYQ_l%f;q0PoQ3Lpxz>@@N<xBhaNW$Smyf$nf=(L{F4OGe}KT>RYTJuB6AL7)^9d2MH^be$HZm!DyzX$iN`5jym<{SMw2!`}ncHvyavBs5o-zSCijHCHW-E+SqtV=YgqfKvmK`T>`5w`q|4eXMNZJnYF&ExU5K!K4{nj-FG7FoF1ZmTkS2?Xt&*{39uwnc)VJof&8wz8g8i)L<JUr4l4~YqSz>mS>4bHA`SA1EC^FTp*~G1KbVz?vp(@5&&8Gk^f5~K__4hZ&&COqfLZsIj{1a=)LpzZCFs`7LLK#(EAs7v`v74kb&FqCs!N6_drv6UZ`z^<YjgCfyXatqBOWPj5R?`IRArbbmMt6y59J@I67k+YlPC8fW}_By#Zmi!wSrXkLHl{ZOxzBo+iYR3UR)__#&omUm_b{Vz?NoIvtXC#<uFep;A3&<gDhNa?WpFFVwJt9Qv)nAS2umnlx~Hmf?|0(F|1*`QBT;i7SjXz_V$GMTqp)s?CE+KMA-Lq<7m?iB8<!@)v#@&^WV2U^DD7Kos-m54TUnP7<MA%I3lgqRYXnWHSltG{nVo4ka8Ya#dI<>C!&_vFHQg(F0t&Y@{tbL-J(e)>qDxGKE{!*(E3IwPor$iK+xC@boR=lk+izPp>3rT)vlgF)J0QM_FF|Y)0Gm-h(=6u2P+DF(BBrwvZ(2^HGzHr?YwXZ^#FEMlhE7{4?x1>38U)fVY*VZ{*w(fzu}gi6-`$-mqsk8yOniucQsbhZddLjs4v;#rUd+`{J+sSVccHKHE3%%Qfo&ntag!+MnkK;B!cRnXxyz~NA%~rh-?t3n;$=*b2O61cU$*Sv`r)|TVWYJeqg)$;d7Brz>u$R=N{4T`cx?O_^U03pZXht)z+h3<XDj#=_1E|C5V9LTAWAv7el{%FA3w)8|OrzYO9ETl5+CG&NTXYcee_qZQ>x9J3`M?AZkFIR|mOev}2>@35Q=A_0$p}4%MO&X(vh>n9+jlh1q8Y!2H2dv3omn)J^1%T}`*-TR;FAB_i@?Dt1k`pR!oy_uz`<^$XW)^1+%VF<2o!4|{b9D+$QbQM}i7BKM-GblJd|F-h853=8Z%v029<E4P>>J^m1xQnKUgQ(gkjoIzmD@nF^GfM%lV)+8O9i7AShw<>DER*cAe=QJ~J<)J5*L?tz|EIo@hfjgVJSQD)RfO<flui`g#Ep+PxmQAGfgV8t0Y<Sshb&S`dJTpHqc`(w_PO~LwOxW?np_oicA!1`~t1*=&KYUD^>gyDwQzy<Kq-HBTCe0&^dt`Z$nuVSSZ;(1Jp%7Zo5zt}RAta<Vj<+Ga=8v)8Ju`F!p@ea>ifEJsu>oe7+pjbJct_`;ZkPxH)-2`E&GbajqT-&IrT@(QPYl*NLSVyTWAA1Dm8jF?(1S;R8#krEL7`dlG*tb@5Q&Fes6tKZTACxPZDlZVGpZ7@nyS&{GmBirBqJA63VN~m8QX3z1&6T|GHNBlc8lax0o_e?kj@q&cCvY(7RX+Aib{o0QZ{8?GOzXU{wyNXEFdP&Nv)Z=z@LnaJ2ETLSayV28g`boQIJGpPbnNlAanI3rFoBKoMR2xVMEl?8?7WNw}!E2P|4F|jN&txxUO38=?cePiPorL5i-PMj!L~<3q;mOn8fEfDn=`Q!eQwd|0(nJ=TwTP6$UvkH*&0)kep8>1emk@oRy<?Jq@oO#vu$@%)85iMqO}6g^WRyAU$@!a%jvfEY=NH+h3c|2H70R7vU}AWKXp%01|^VIj5y@@vn<b@6P>{7@4YgLv6W9g2c^zgMC7#4+6Q1E5T2j><gO3kwg>J7Te2IMz1#n^2D7S42t1%X`;6|FwTvB;m+VRad5?u4L~jelM#ho(=x!OE`8DxK)>_2A`G3>?nz$LFR!i`c-WBG0$Vq=QvvG5(WO<lAi8?4=eW4(G$^1Rs|`#X><tD-$xC}3Z{xLi@ubH&eYBCaMB;Xo2_S6rqdjnOQ_`fS3<gH@W1J>=;|3j)OQW!ucm__l+%~$-SZ^>>8k+CSZFJ@=%G(cc^w?_PJ1^Xg3KNSv3?&z&fOe<}M&MkC7_zz>SK?;0xFcc-0UnM;hx+W0RIgLwo(x7hPtWF<5Xq?x@vZ90pic1RstOo!6iEk<W_#oER6LK|#Q5A1|2sM3<Ka0Hx2NM%)q_kJU!*G|24IemBvY;3>=Ky@Kqqom#i4>Xw0qF|f;N=!E?L=lE^P?xq;&K~C;Ekf9`j}>qEj3ZdIRZ5LZkJn5%Pwkeb4llHJcH__y8i_EODTOpommaq<QeJl{hU5y9Pk}H|sX3ar0X>HhAQ_3twx?XH*|a4S?kWyqRcMr&g#|H5td8a%lM96bXjuxq@;*MfZW^mEKUn`X>rnq*}nP2Bx!GDMm9dVya>kKG@}8170<g-r|0IdUGYW9!C;N$egnO@(W<QSyvS!&{_O>`Y+k$v9|-0rObOB+jGF3R!3BhRS(3Yp|ryRHdxsNrHQ5E$y!hc?2n47a-D4EIBRqO9{rLPBT<aQi_`I3C7{P3N({+GL8!oTwVcgd%MTGr!_ldbVF;7rkhGrEI4~rdwYo~%Q@veQvn63SOjS8UR%%%S&A|=yCj)&th&v>aP+KRgJ_d?>2It_~=oDy=0};b15zbg!fDOQm*OwM+XQBp(=YmF<2&9uoSr4vmhf93<4r<qp)Vsh=*O^&8EggwqH6u^O)nq%>A!|#45pbfR_h&AGH5nDZb%;O<neaB;h07a5Xo2^MCUhkH#}#mfQD9hsJ53@)h)3Zm<X52)_d<rBiPgbzUWqk%F!~bk!AkNr@`UDHk}|HMFJdCEjIuoKHsE35Y=sHV6i?J~+AyGcE3O=xLV`RhY=#+Ir81>%#(vwcd_?M_Is9!kAE;71O<oVW=v8{cRBimExqYjFJ3W;lGO^J(&_a((G3(d+M|6Hg7xjRL0_kuZSxO59P&PoOFHwX8#1@hv#*@mPL76@{npK>j=6qty)N2Z#W~q(Q0n*jfau0!_Sp#`md<>A$v*~j>d>%plL6pxh+ME<|2UYpe(#sgNsOBUMF6wiG0f89Js3yf5QzY21WJALVD^jUcltaxs@|tiCu(GCEbX&T@nYFSd2+ZjgGwOsyIk9Y9##~y78-<!`#>yur@=QNxQih_|V{K^kqKRk;tB~eWoupu^IBUdui9}?iXH*h%9!-g$>=>!jWMul$+bu7ZKNjKtdYU3X_=G|n|3j{bA6??&<XLlUpWfM*-dX-0)NbyirG{2QA!=mw1X<#7oaaNkPQ`?})_6~nO)wd(IztQfgJLnTWFYRMWy!AWv1!kqHI8O+HowRGl9WU#_xFoc|1|As*eaEkStRav8|oguXAq-@q=bPMiI_Q8SeSrYAqh}SM=FpMRlgPr0~{|z+@4{QJ5&TToqummSQ5$W+N?&)4vC3itis5F%(Fn)g;4h&MhVxiNFirW<D^g7Y`oswlnlK^`I-6Fvg30_sD-dzq*idTR=9Fz3eLvEXc*feTSTSTT4|{8E5qUwS``NIE16GTj5JfK)H`fDy40$K`mOOUelR0h2Xy@lkxyMn1KOy&+Pf?1Mdv}5%2y^oWRvYYv4>EYL0zg&3FnW2G1W_jzufMeWADGk(wzUa-Zc+z1a}NQbe4CuP<w>&KILCh4T#BfD)EI^R$~fydWBMoM=JX1v_?+zJtH&j<58opl(RscB!WJv#<j*o(K^SUW}Q|}->yIB2x_W_VZFaoLZ6o9w7ztlC3bka(rTcj2r0Bm)e5m<3628^i7seid_3(_ddflG&T8B>n=^}+$$wzGwc?bjIfybQti=%+nc_-~_pIZcaw>F|y)s0%A3;>ktM;~&y-JS|-tC;3KjK1kVbhXD)zGuh@{v~5ZKQ!R=sU~3%o55Sct7|s6=1szg5BkZMUy@$6?*#EN!id>^a(%#l)fTxt$UZMevhO-8~JI4Vtg9bb8&zLXESfHZP&px$rLKO*G>nW&c5X^i`{IxC}}b(isv*V9s!ZGb!qf{_@?ULz<#-!`MHZ?P3Qs>bv_8w7yJjPC=?%;UOuQ;jvNCRNqQLpA^v`3WlkoP7nKcTqOD(_l9<p$Ykm@LNi}-^ULZiOz{kjw+-k2q0b&`W0Nf?45x<NR?@8U7S+U?wZON_kRHEIg%dnK>iiO;*>3^J%?<E_a!7bMeiD8G0)Dkah4~mh5gvf|J)JllhXAPh=Rom6{LL0c$X5X^EAAb7$i+`=C4Hiab=hAJzJ4muYZTt!IYc7(G9yCu_!{-XfC;`7rZl(!cp}(qsR9z-DwOZ$2Z~k3y)>3}f>nGCsw-wSlsXP*9KZ*mlkCZUAiU&2K$X+#e6PdS_v^w^|x^i^UZQukJ(vR|^4wFK`1?&BWS+(c)rIVtdXH;tq_Rx-#vj8X};b#qsC!Q(D4bmTdd)o4R`6XV)4p<jAC8<`*GZ7iWoP28<oW0*b%3dYJ71pUAcCt+GM+wm><=kWmubj<lnOvqdQzr2-#v_r*&C?d@3_aA+f%1Ml93<x>T7>uo9`C`F7^azuR&KwgaQlHg62YZB^kRYxU_H4`jC;Rb8@LEfh@-QEfqJTzqPAdbv|w!jpkFV>@3{15;g9u26i^K9$;8&e`CaK4QSfRKKAOa_9Qm$T^-BFHk~?8oEr?OZu<%j160PA!?8X8d4qo(71?Mj7NZDZ|4F4wj2gj?@@<5;mlzdqQbpkrw7KO#u{X1C`P5UhXd%JDpeU3<5icu{#CvTw1`naT48UZq_;2<U9fYBYTzICYPzl`N1!9jnMb%!ND;kqii|2@6iUSmxLOVO0d*mAm8z;($hK=nHeQ^FJ<78{t`K-yG<XGfKq>l#Z$wMtoavy{}fTd|V9<7J_oJMFrX^i{Gt5NjY({xH2H2aUK4QoHI6$M$4S-$}G1SeEZq9(B}@J#<whOiG@_WOeH-SX*83T8U`Vn{t1{L(>8?lNIyYkBW^bXhxvhgPf{gI!7R^`l>xRYF=q7N!$kB<&%TyJGQ|GAsE^XM>T-uJ;ZfQF4-nU<-QmU&BGuV?Tpmf!2&!>IB5#LB5M6d&xySY({6=plKR~=C0o~gp@7nEI`$&L*zRdWsC%f@jx_mYK>9?)M>?*S;c(I=Y1SW<oJDmj`>t0(>QtH8vd_h1UZ13j%H@im<myItq%El<Cz#h$VtX5e-ffOyQ1shcliKtLk*Yvq-ub{*yt!_rbZo&t8VUqTcB8a$hiS0J1|~fX_nZz0GduWBAH=W_FwF*igrcjO;xh1PRoN@YP(ia>QhZD|z8Faq+ziUzqcJ_?TDaPt{avi!XppV1tmJ;-v?m$1KCHE$_esbWwYXvEqRF@<2!MfYsg4uvBh3DWRU+7$A0C4h-%3D!rNjxw>Z&+MEY=QokuCt$fsr6I4R24wBm<!kbAscu1U}<};-W&NNDqiBgOM{$iRSHq367T?4<W!fW|<fgAqg>wM;d{6hdjI#)s@UXO=N2ZI$&a5XDwU|mfPJ`{dlwv#u7hq{L(Orp{Mnub?BUUiBkF@+nx04#1&fC&~vpTHrsBB8uHKst$pqwG$Y`&F>wj@CL0}BP+|tX0o@?cnj4)SVRbdXkc&QO7~Bw=q!CfCM$Ed1tv2QYnGddnkrid2x)w>q3u~S~N0~=UmJ<hg{UH(S9;byB9<C5`TBiz-r(x9zFpk5kL&$(g8dI{di<_>tS4)Z7Fu{p~fnmL{MVsAjE9A#B<)sC1-Z5sgQIAhQN5@LHGiEJFNzlbA<Fy^dr9TB<l5Bxw5G;ChQTeN&GM-wMWbHEnldQ;>O}6V!Y1Lv8EI`g{yFqZqbv!+MieX_BIMh0GUR9pcRH{QnsY&m<U#O03loC~IG49-qY818`e%K~#(^$4NTb-0;3Uu*%Tk`}qEc7(q^s3YvY(Sj&{A4Z8>PoHRn`k4X1~{-5q>Jh1WO=H~2nUq9O%2=;G28r{5FhI51+MCOYJ2nHXNV*?$T1SOZHJtRQX@tyvX^#jqKmo5R;D@n72UZhz2+)7TP{265v);CBsfsW%PAYJ2WZdjB$JM)+Tg%=T0M1LCqQcV7eL`iMX&uG-mBJroIfj0DSW<oz6$4$F2R;{8nLE<Oay&2fF+s|Wn3h!_pgLesZF6uJBb6)MW3N3?c+K`y_b_-*JFjO)d(MN9O0}>gq(JdvM_X@<~^mN=U2NN+#Z)<W>K*<tM;Y*$ssy0*G>4otW^Q2ViEdD7RBRQ(Q1q5rjLh5_CH9Ts2<wxgbTbaMe-SADUb59#SAuk3Pu`*YJDvgmvI$;4<yDCjJroD%{iJgCXwi)MGFr`)1-CuE|w1LBk>G5H#>(}_dU&_Eli@Lj!6%pD`Z7C?daQYX;8Y+mUH_3mE{~F6WT84I=?N|me8ahJHa}AHO9^0h)A-51Ur>9D)PqkQ!SxJLDLZad(h?Wbba&UI~oguyx{%RoUAKdGo+)JF2T(0@9t-#7}>-$>0>D6?rY8$6~~;s@mzOO8lnk!8D5<RAQ8Mlx#;6f>0bNERS04NEUSAciiGgiefa5qI&8Y)RgKLjYI?Br1c8^goVP$r*7pBO1nT;4=tsazm#fek{qb9KAK5@M?ajUJXw6-EFD8(t0?sOvQ0TuTE}B__$cRcqnsZBJNEw_txoIUOi)MY>nt@xHu(0gfD|-X^)}{Ko6$Nz)!&RwSw^2Z%OcNH6HburE%^GIkNlWY$mQ!Ch&E09L!60Vg>`qH>4T=%5wTpRzW5nL1ZriLw^2%MsRMV(ye|Chvd~@c~b!j7VNjt6BXGGgJbF~Shk`(rs4(|3B^z|iA19b&s*06M;w-uv^Nu#APq+J>MTTlQrHpWS{$HplB(5wtsRY<keE57*Ep=&SWQtJCm%OI`LQ)RI(2Lb`?TQsFmr$rDT)(Kry+?Q};So`*(w;{DC2THKsmLxZUQbZtOs8Az0Et&lUivvmL%TNR?Lo~g<*1TxhvN9V4gi1h06I+yuEwcerS4RSU8pKB*;O)@OYx6CJzR$Qtxy&PuuAntZKuI5VR`=F0q%1P3$HcMluu?Ew@6}Kga|QCJBkI?&RQ(6!)0;Zno~Z3vWsNG%y*Vl~wh5ZDNOMf`X&Rd?%LbEM%7}!8O;X47M%n$YgGcizTUc3_vmr;AQ7UoJtu&bRw?I?t#7uIZE7!U!v+jcX01bYaltA84r^W}%66gnL^LU`qgMKGMO=F~K&O6Rav5b-JR`>)%<>6)jnqCDZ;iVJ{u1Z^5VhRR;uE4Xa*#Y%XAAwS&ztU7quduY2rRHk3Hj)Y<MOlVull8D0+rSY#s0&7^vOUrmL|!sNm-G$Wlv|Go94PF%4~%c`xxLaErum}_dZ=UA{xnslg&f7Om>wYW`}5Qb#wxcNQ{bMwb%bd&55KL&Lnwg+&DYttrD{Omf;K^y29szRv&NMwE@ksEx`(`e*;!M6zN=l$Oodt0StUH{*h4uW1(E`cCa2fo`9jYz0%uY_vfwR+QyHgoJuj_+H1t3wr8}7|{QW0@hp62&*`a~a*R6gTw8;<f6j=;KEn~`I(FFPO@o>zfa+&Zdx1twH4~kTW$|wh^Y$B}oy6;WjVugEEkRij}ElQvJ@gt(xtxdV;W0V&q-($_w0*3d}JW%Z<?MWc1?1-JwDkUS|?ut*bwo_QTAbo;m#Jnt-5>tR7gMS%JXzgDX%4Pk7N}tkxcM^89j3_CrG_^bR5Hk*CDrbuxo0ZV6+aV-n#q*aw`tT*CdSV03&lQ(nK@clGK^S$i=Dwmx2{3+pl2qsqsJ$uhCY4dB)l|!O)6A7Sm&8R5U|>c|azJHA<1<0;Nt5HG;N^tSg||^cEX@w!_?{c7>F)`+?LG2Q;@aCG1+YGqU39`eZ?<crC6@rW;op%fKgz_nj)39-NC&N4(_yQ}WuCxp^blK`qG!=L$XesMFeD*X%Dwd4Qk5>7VX2!WsT_#s%;or=_&%75(&+A9zaJ48{1a+2hP}BXdWO3+86lNQ)lVG?LnWrQtSW-X9)N{bqpw8GOS_^R)05%6xB>89ua}GZ5C|+Zl-*;&czPkc(^@W-T$j5JX{VmloGrs~#SLMGbq_tDb`q|8FLNIOQgD@G)<X%(aUUV7xKXYkNnYyRB=7icC48q%%;Cr|ZBCNdP1N&b4T>RJ?BI!?Dbuouv!ZYa7IjcDlt6Qm0ivi}Zr4D^)~>~k)o3jl-mfPY3;;(4?>b_&c+eSGx3EcDvMcmTgd1c@(fzVzoQeb<me>_*lUi&gOajkQ?c5;oszTAI`^X6DZ!PU2-l+(0Y7mOVZANg9T9Z($s`n;xr`oYg_bu0E0?+-cK2~l=mn6EUp?3wjx8cGD?w3lF)Z&D9W`KZ)CocellOx`&Ow<_TEw_mZ-+{MtUcB(=j3L54AgaAsKH;oRtejok6){ChH%9Xc;^(r0DbczRYL)7G77G~t&AHKSaKB?#PoOTiKv3X<w&gDRy@v@;)?hoAEt4s;jsKmbhk1*>tm$$Vs20a!*9>2yQft0>$YCZbYJ>r7ll*?);IFn@agF4f^URU~ClcPmE4@gU1peKlUV0Bq3DTVcNvip?<+ZvN;Dp*1B*M@Dpzu5&j`F)Ml_u5$!{qML8<2Gw63EbEb+iIw-S;7t2m>oA>ouAVz~^-aC_BcfR+MWfnM;V2g7wJ_PAHa=afjM0CkCX%qCpY?GGvq_6|?jh$oC2@%mctKNqd9G6VC`NDgl0apwesJgV?n_u1iO$=Eh>^RH}xiQqS&f;w#L2l2OJ59ip4Ix9S5r>&-*tz@e!+l!YMRz<w!(2ZwLvqdMIE{gSly(PlQ|p}mx^Z*8>y)5wCjUJFc25kOAWD6;!Y*_%|RK`@~Tj~9v5oVDu_Z(uNl<lKmT8htJ3Y5px-h$ziUM;H=g*`eQo9m-_2TdU&fzPmRsPV`8@`wlM!T%}`wWJ_MDi&PhJ>OdlU+gJR-m>OqK^X6(lEEnX3)0hwrT>+Dl0Vf?ul|Imp)IuK32qWa#Gu3Tnl?{)Dv04&hB?D!0b%314wo&yG1e5#ZF+FKNskeTq0NS}_%$+tPDUFzVm54y!YLkR-EiAEtT4tcWYUDW6OvqCDE-CVARcE6mmD4%sF5B+}_ev|uU}`Zt1^&V#fwzFRPh@Be8}-K9)b!~P?o6T~xveO-<vESi9irQO7Rb3N{G>Itge1hy-P=|Z7`~w~tHTGBHCEcNrS}KXrnGHzmQ_G9%TH&qDN{5!D}mABTVfQT2qJ;XcYE363U;q5N6@D46dUz!cY5b-O&{ihAum9PLP~zIZG7o?CTG1vBEz;N3_G-*dQ+>0-3cHoU#pm+f11k@jA^*oTJ$)y>WC^AiGp8L=bb9ZrSbD=(V@F0tk8gJLA!WDT`|J9T2w%ncJquulSZ?v93ka2v&@<_wMrtKPl!;c223Ef6!}6H0JL<v;#g#$=kh04&;n_<L-mo=nR5h(#K%;1)m(Cw*P*Sxmnmh6+?o(kZEf{dVN*W7wxnUG>$-dxI}F^adn5*Xe-r`%R(kTWbUc4<=a3}v+{oheG|jGwNyn8vr>{RxP_0HgUG=)PXcSkoGCqA+bTrngrsLGY`ovEN5031Y-l<H{y9`!H3Nw`z2Y0mQGQzTpq~XCDAdIe(Qm~{sh?Hv560((v9ratGDt5MMXSMB7Bb=nP!574RvGE^NhQ@6W(=W4r=-+i?$5R^^p;kQMsOIrx3HaMC6$!7P#<h)<jNI%vQOqLD12g}v^ed<8Dp@_(JzR4JK;Hl)4r^9Ll<nSYE}h`9qOp^T*PT8#z_%ZMNRu13&t?z8gpYOjp?10LGnzky+TWJ)m7_AiH!r^S&&MQ=iR5{fkN1orGjgKm$H#Ac<ZlF^0J}w>^gBbcCI;qpcXaZd`B66RLP7p^stYBQ3vfPW4?yiYxwvh+$oZAtox4-PgytkZywd&<y~m==gJMlc`Fi>T<@l{{|M%wg-+%e(<=Lla*Q?cHyIgMn<v-uNe)aYu{_*nVUw(M~>gWGD`}Ch@>x;{GpZ4xkK3%>0y!B>zcwzDGh0W#a@WSQo3!B3W7iZ^_-)t8TZ{6I!usQzQ?c%ci+q<`3&)&LRL~mVRy!*}FT)cnbVa)5>7uM5v{Px>#Z|CEu|9JVgAAk7&UjFjtuW#q!)8*TjUjF>@r(a+H@@^vj^77w*{QJ+p{_wwk`SI0H|M7O(ep+07@@D<Xn_qwV@gM8uCx3hS^XpfO<=cV0{PCBc|Lcc0zrOr`&putO*XL*d{>#gk|L5i1E6c_9x8MGMD0Z{v')).decode('utf-8')
)
# END GENERATED ROUTES
_BRANCHES = _PAYLOAD["branches"]
_DEFAULT_BRANCH = "c10-s4-straw42-melon12"
_EARLY_YARN_BRANCH = "c6-s12-straw42-melon12"
_LATE_YARN_BRANCH = "c6-s8-straw42-melon12"
_LOW_DEMAND_BRANCH = "c8-s6-straw42-melon12"
_MILK_DEMAND_SHOPS = {"PIZZA_SHOP", "ICE_CREAM_SHOP", "SMOOTHIE_SHOP"}
_ACTIONS = _BRANCHES[_DEFAULT_BRANCH]["actions"]
_WEED_ONLY = _BRANCHES[_DEFAULT_BRANCH]["weed_only"]
_ENABLE_FIELD_GUARDS = True
_ENABLE_PURCHASE_RECOVERY = False
_ENABLE_SALE_CAP = False
_ENABLE_FRONT_RUN = False
_STATE = {
    0: {"last_step": -1, "active": {}, "branch": _DEFAULT_BRANCH},
    1: {"last_step": -1, "active": {}, "branch": _DEFAULT_BRANCH},
}
_WEED_REPLAY_STEPS = 8
_PREMIUM_ITEMS = ("MELON", "MILK", "STRAWBERRY", "WOOL")
_PRODUCTS = {
    "CARROT",
    "EGG",
    "FERTILIZER",
    "MELON",
    "MILK",
    "STRAWBERRY",
    "TOMATO",
    "WHEAT",
    "WOOL",
}
_SEED_COSTS = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}
_ANIMAL_COSTS = {"COW": 400, "SHEEP": 500}
_LAND_COSTS = (1000, 2000, 4000)
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
_PURCHASE_PRIORITY = (
    "WHEAT_PRODUCT",
    "COW",
    "SHEEP",
    "HIRE",
    "WHEAT_SEED",
    "MELON_SEED",
    "STRAWBERRY_SEED",
    "CARROT_SEED",
    "TOMATO_SEED",
    "LAND",
)


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _shop_list(obs):
    town = _get(obs, "town", None)
    shops = _get(town, "unlocked_shops", None) if town is not None else None
    if not isinstance(shops, (list, tuple)):
        return None
    return list(shops)


def _select_branch(obs, state, step):
    if state.get("branch_frozen"):
        return state.get("branch", _DEFAULT_BRANCH)
    shops = _shop_list(obs)
    if step >= 144 and "early_yarn" not in state:
        state["early_yarn"] = bool(
            shops is not None and "YARN_STORE" in shops[:2]
        )
    if state.get("early_yarn"):
        state["branch"] = _EARLY_YARN_BRANCH
    else:
        state.setdefault("branch", _DEFAULT_BRANCH)
    if step >= 216:
        if state.get("early_yarn"):
            branch = _EARLY_YARN_BRANCH
        elif shops is None:
            branch = _DEFAULT_BRANCH
        elif "YARN_STORE" in shops:
            branch = _LATE_YARN_BRANCH
        elif _MILK_DEMAND_SHOPS.intersection(shops):
            branch = _DEFAULT_BRANCH
        else:
            branch = _LOW_DEMAND_BRANCH
        state["branch"] = branch
        state["branch_frozen"] = True
    return state["branch"]


def _copy_action(action):
    action = copy.deepcopy(action or {})
    return {
        "farmer": list(action.get("farmer") or ["PASS"]),
        "hands": [
            list(order or ["PASS"]) for order in (action.get("hands") or [])
        ],
        "market": [
            list(order) for order in (action.get("market") or [])
        ],
    }


def _seat(obs):
    return 1 if int(_get(obs, "player", 0) or 0) == 1 else 0


def _farm(obs, seat=None):
    seat = _seat(obs) if seat is None else seat
    farms = list(_get(obs, "farms", []) or [])
    return farms[seat] if seat < len(farms) else {}


def _branch_data(branch=None):
    return _BRANCHES.get(branch, _BRANCHES[_DEFAULT_BRANCH])


def _route_action(step, branch=None):
    actions = _branch_data(branch)["actions"]
    index = min(max(0, int(step)), len(actions) - 1)
    return _copy_action(actions[index])


def _weed_annotations(step, branch=None):
    annotations = _branch_data(branch)["weed_only"]
    return set(annotations.get(str(int(step)), []))


def _align_hands(action, obs):
    action = _copy_action(action)
    expected = len(_get(_farm(obs), "hands", []) or [])
    hands = list(action.get("hands") or [])
    if len(hands) < expected:
        hands.extend([["PASS"] for _ in range(expected - len(hands))])
    action["hands"] = [
        list(order or ["PASS"]) for order in hands[:expected]
    ]
    return action


def _reset_if_needed(obs, step):
    seat = _seat(obs)
    state = _STATE[seat]
    if step == 0 or step < int(state.get("last_step", -1)):
        state = {
            "last_step": step,
            "active": {},
            "branch": _DEFAULT_BRANCH,
        }
        _STATE[seat] = state
    state["last_step"] = step
    return state


def _tile_at(farm, position):
    try:
        x, y = int(position[0]), int(position[1])
        return (_get(farm, "tiles", []) or [])[y][x]
    except (IndexError, TypeError, ValueError):
        return "LOCKED"


def _positions(obs):
    farm = _farm(obs)
    return [
        _get(farm, "farmer"),
        *list(_get(farm, "hands", []) or []),
    ]


def _actor_positions(obs):
    return _positions(obs)


def _inventories(obs):
    private = _get(obs, "private", {}) or {}
    return list(_get(private, "inventories", []) or [])


def _actor_label(index):
    return "farmer" if index == 0 else "hand:" + str(index - 1)


def _is_weed(tile):
    return isinstance(tile, dict) and _get(tile, "kind") == "WEED"


def _has_animal(tile):
    if not isinstance(tile, dict):
        return False
    kind = str(_get(tile, "kind", "") or "")
    return bool(_get(tile, "animal")) or kind in {
        "ANIMAL",
        "CHICKEN",
        "COW",
        "SHEEP",
    }


def _inventory_count(inventory, item):
    try:
        return max(0, int(_get(inventory or {}, item, 0) or 0))
    except (TypeError, ValueError):
        return 0


def _field_order_is_valid(order, tile, inventory):
    if not isinstance(order, list) or not order:
        return False
    verb = order[0]
    if verb == "CARE":
        return _has_animal(tile) and not bool(_get(tile, "cared_today", False))
    if verb == "FEED":
        return (
            _has_animal(tile)
            and not bool(_get(tile, "fed_today", False))
            and _inventory_count(inventory, "WHEAT") > 0
        )
    if verb == "COLLECT_FERTILIZER":
        return _has_animal(tile) and int(
            _get(tile, "fertilizer_available", 0) or 0
        ) > 0
    if verb == "HARVEST":
        return isinstance(tile, dict) and int(
            _get(tile, "yield_units", 0) or 0
        ) > 0
    if verb == "FERTILIZE":
        return (
            isinstance(tile, dict)
            and str(_get(tile, "kind", "") or "")
            not in {"", "WEED", "PASTURE", "ANIMAL"}
            and _inventory_count(inventory, "FERTILIZER") > 0
        )
    return True


def _guard_field_actions(obs, action, step, weed_only):
    action = _align_hands(action, obs)
    farm = _farm(obs)
    positions = _positions(obs)
    inventories = _inventories(obs)
    orders = [action["farmer"], *action["hands"]]
    guarded = []
    weed_only = set(weed_only or [])

    for index, order in enumerate(orders):
        tile = _tile_at(farm, positions[index]) if index < len(positions) else None
        inventory = inventories[index] if index < len(inventories) else {}
        label = _actor_label(index)
        candidate = list(order or ["PASS"])
        if candidate[0] == "DIG" and label in weed_only and not _is_weed(tile):
            candidate = ["PASS"]
        elif candidate[0] in {
            "CARE",
            "FEED",
            "COLLECT_FERTILIZER",
            "HARVEST",
            "FERTILIZE",
        } and not _field_order_is_valid(candidate, tile, inventory):
            candidate = ["PASS"]
        guarded.append(candidate)

    action["farmer"] = guarded[0] if guarded else ["PASS"]
    action["hands"] = guarded[1:]
    return _align_hands(action, obs)


def _trace_actor_action(step, actor, branch=None):
    actions = _branch_data(branch)["actions"]
    trace = actions[min(max(int(step), 0), len(actions) - 1)] or {}
    if actor == "farmer":
        return list(trace.get("farmer") or ["PASS"])
    hands = trace.get("hands", []) or []
    return list(hands[actor] if actor < len(hands) else ["PASS"])


def _repair_live_weed(obs, action, step, state, branch=None):
    action = _align_hands(action, obs)
    farm = _farm(obs)
    positions = _positions(obs)
    unit_actions = [action["farmer"], *action["hands"]]
    active = state.setdefault("active", {})

    for actor, transaction in list(active.items()):
        index = 0 if actor == "farmer" else int(actor) + 1
        if index >= len(unit_actions):
            active.pop(actor, None)
            continue
        age = step - int(transaction["start"])
        if age == 1:
            unit_actions[index] = list(transaction["intended"])
        elif 2 <= age <= 1 + _WEED_REPLAY_STEPS:
            unit_actions[index] = _trace_actor_action(step - 1, actor, branch)
        else:
            active.pop(actor, None)

    for index, (position, intended) in enumerate(zip(positions, unit_actions)):
        actor = "farmer" if index == 0 else index - 1
        if actor in active or not intended:
            continue
        if intended[0] not in {"BUILD_PASTURE", "PLANT"}:
            continue
        if not _is_weed(_tile_at(farm, position)):
            continue
        active[actor] = {"start": step, "intended": list(intended)}
        unit_actions[index] = ["DIG"]

    action["farmer"] = unit_actions[0] if unit_actions else ["PASS"]
    action["hands"] = unit_actions[1:]
    return _align_hands(action, obs)


def _weed_repair_action(obs, action, step, state=None, branch=None):
    if state is None:
        state = _reset_if_needed(obs, step)
    return _repair_live_weed(obs, action, step, state, branch)


def _unit_orders(action):
    return [
        action.get("farmer", ["PASS"]),
        *list(action.get("hands") or []),
    ]


def _order_quantity(order):
    try:
        return max(0, int(order[2])) if len(order) >= 3 else 1
    except (TypeError, ValueError):
        return 0


def _same_turn_deposits(action, obs):
    deposits = {}
    inventories = _inventories(obs)
    for index, order in enumerate(_unit_orders(action)):
        if not isinstance(order, (list, tuple)) or len(order) < 2:
            continue
        if order[0] not in {"PLACE", "DROP"}:
            continue
        inventory = inventories[index] if index < len(inventories) else {}
        if order[0] == "DROP":
            for item, raw_quantity in dict(inventory or {}).items():
                quantity = _inventory_count(inventory, item)
                if quantity > 0:
                    deposits[item] = deposits.get(item, 0) + quantity
            continue
        item = str(order[1])
        if item not in _PRODUCTS:
            continue
        quantity = min(
            _order_quantity(order), _inventory_count(inventory, item)
        )
        if quantity > 0:
            deposits[item] = deposits.get(item, 0) + quantity
    return deposits


def _pickup_reserves(action):
    reserves = {}
    for order in _unit_orders(action):
        if not isinstance(order, (list, tuple)) or len(order) < 2:
            continue
        if order[0] != "PICKUP":
            continue
        item = str(order[1])
        quantity = _order_quantity(order)
        if quantity > 0:
            reserves[item] = reserves.get(item, 0) + quantity
    return reserves


def _existing_sales(action):
    sales = {}
    for order in action.get("market") or []:
        if not isinstance(order, (list, tuple)) or len(order) < 3:
            continue
        if order[0] != "SELL":
            continue
        item = str(order[1])
        sales[item] = sales.get(item, 0) + _order_quantity(order)
    return sales


def _cap_sales(action, obs):
    action = _copy_action(action)
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    deposits = _same_turn_deposits(action, obs)
    reserves = _pickup_reserves(action)
    available = {
        item: max(
            0,
            _inventory_count(shed, item)
            + int(deposits.get(item, 0))
            - int(reserves.get(item, 0)),
        )
        for item in _PRODUCTS
    }
    market = []
    for raw_order in action.get("market") or []:
        order = list(raw_order)
        if len(order) >= 3 and order[0] == "SELL":
            item = str(order[1])
            quantity = min(_order_quantity(order), available.get(item, 0))
            if quantity <= 0:
                continue
            order[2] = quantity
            available[item] = available.get(item, 0) - quantity
        market.append(order)
    action["market"] = market[:10]
    return action


def _purchase_key(order):
    if not isinstance(order, (list, tuple)) or not order:
        return None
    if order[0] == "HIRE":
        return "HIRE"
    if order[0] == "BUY_LAND":
        return "LAND"
    if len(order) < 3:
        return None
    if order[0] == "BUY_ANIMAL" and order[1] in _ANIMAL_COSTS:
        return str(order[1])
    if order[0] == "BUY_SEED" and order[1] in _SEED_COSTS:
        return str(order[1]) + "_SEED"
    if order[0] == "BUY_PRODUCT" and order[1] == "WHEAT":
        return "WHEAT_PRODUCT"
    return None


def _purchase_order(key, quantity):
    quantity = max(0, int(quantity))
    if key == "HIRE":
        return ["HIRE"]
    if key == "LAND":
        return ["BUY_LAND"]
    if key in _ANIMAL_COSTS:
        return ["BUY_ANIMAL", key, quantity]
    if key.endswith("_SEED"):
        return ["BUY_SEED", key[:-5], quantity]
    if key == "WHEAT_PRODUCT":
        return ["BUY_PRODUCT", "WHEAT", quantity]
    return None


def _build_route_targets(actions):
    running = {}
    targets = []
    for action in actions:
        for order in action.get("market") or []:
            key = _purchase_key(order)
            if key is None:
                continue
            running[key] = running.get(key, 0) + _order_quantity(order)
        targets.append(dict(running))
    return targets


_ROUTE_TARGETS = {
    branch: _build_route_targets(data["actions"])
    for branch, data in _BRANCHES.items()
}


def _count_animals(obs, animal):
    farm = _farm(obs)
    count = _inventory_count(
        _get(_get(obs, "private", {}) or {}, "shed", {}) or {}, animal
    )
    for inventory in _inventories(obs):
        count += _inventory_count(inventory, animal)
    for row in _get(farm, "tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and _get(tile, "animal") == animal:
                count += 1
    return count


def _visible_holding(obs, key):
    private = _get(obs, "private", {}) or {}
    farm = _farm(obs)
    if key == "HIRE":
        return max(0, int(_get(farm, "hires_today", 0) or 0))
    if key == "LAND":
        quadrants = list(_get(farm, "unlocked_quadrants", []) or [])
        return max(0, len(quadrants) - 1)
    if key in _ANIMAL_COSTS:
        return _count_animals(obs, key)
    if key.endswith("_SEED"):
        return _inventory_count(
            _get(private, "seeds", {}) or {}, key[:-5]
        )
    if key == "WHEAT_PRODUCT":
        total = _inventory_count(_get(private, "shed", {}) or {}, "WHEAT")
        return total + sum(
            _inventory_count(inventory, "WHEAT")
            for inventory in _inventories(obs)
        )
    return 0


def _purchase_state(obs, step):
    episode = _reset_if_needed(obs, step)
    return episode.setdefault(
        "purchase",
        {"purchased": {}, "pending": {}, "awaiting": []},
    )


def _reconcile_purchases(obs, state):
    awaiting = list(state.pop("awaiting", []) or [])
    purchased = state.setdefault("purchased", {})
    pending = state.setdefault("pending", {})
    for attempt in awaiting:
        key = str(attempt["key"])
        quantity = max(0, int(attempt["quantity"]))
        before = max(0, int(attempt["before"]))
        after = _visible_holding(obs, key)
        gain = max(0, after - before)
        if key == "HIRE" and after < before:
            gain = after
        fulfilled = min(quantity, gain)
        if fulfilled:
            purchased[key] = purchased.get(key, 0) + fulfilled
        missing = quantity - fulfilled
        if missing:
            pending[key] = pending.get(key, 0) + missing


def _record_purchase_attempts(action, obs, state):
    grouped = {}
    for order in action.get("market") or []:
        key = _purchase_key(order)
        if key is not None:
            grouped[key] = grouped.get(key, 0) + _order_quantity(order)
    state["awaiting"] = [
        {
            "key": key,
            "quantity": quantity,
            "before": _visible_holding(obs, key),
        }
        for key, quantity in grouped.items()
    ]


def _fib(index):
    left, right = 1, 1
    for _ in range(max(0, int(index))):
        left, right = right, left + right
    return left


def _unit_purchase_cost(key, obs, offset=0):
    if key in _ANIMAL_COSTS:
        return _ANIMAL_COSTS[key]
    if key.endswith("_SEED"):
        return _SEED_COSTS.get(key[:-5], 10 ** 9)
    if key == "WHEAT_PRODUCT":
        market = _get(obs, "market", {}) or {}
        prices = _get(market, "prices", {}) or {}
        return max(0, int(_get(prices, "WHEAT", 0) or 0))
    if key == "HIRE":
        already = int(_get(_farm(obs), "hires_today", 0) or 0)
        return _fib(already + offset)
    if key == "LAND":
        unlocked = len(
            list(_get(_farm(obs), "unlocked_quadrants", []) or ["NW"])
        )
        index = min(max(0, unlocked - 1 + offset), len(_LAND_COSTS) - 1)
        return _LAND_COSTS[index]
    return 10 ** 9


def _estimated_order_cost(order, obs):
    key = _purchase_key(order)
    if key is None:
        return 0
    quantity = _order_quantity(order)
    return sum(_unit_purchase_cost(key, obs, offset) for offset in range(quantity))


def _affordable_quantity(key, wanted, obs, money):
    total = 0
    affordable = 0
    for offset in range(max(0, int(wanted))):
        total += _unit_purchase_cost(key, obs, offset)
        if total > money:
            break
        affordable += 1
    return affordable


def _recover_purchases(action, obs, state, targets):
    action = _copy_action(action)
    market = list(action.get("market") or [])[:10]
    slots = 10 - len(market)
    if slots <= 0:
        action["market"] = market
        return action

    money = max(0, int(_get(_farm(obs), "money", 0) or 0))
    money -= sum(_estimated_order_cost(order, obs) for order in market)
    money = max(0, money)
    purchased = state.setdefault("purchased", {})
    pending = state.setdefault("pending", {})
    scheduled = {}
    for order in market:
        key = _purchase_key(order)
        if key is not None:
            scheduled[key] = scheduled.get(key, 0) + _order_quantity(order)

    for key in _PURCHASE_PRIORITY:
        if slots <= 0:
            break
        target = max(0, int(targets.get(key, 0)))
        remaining = max(
            0,
            target
            - int(purchased.get(key, 0))
            - int(scheduled.get(key, 0)),
        )
        wanted = min(max(0, int(pending.get(key, 0))), remaining)
        if wanted <= 0:
            continue
        if key in {"HIRE", "LAND"}:
            wanted = min(wanted, slots)
        quantity = _affordable_quantity(key, wanted, obs, money)
        if quantity <= 0:
            continue
        if key in {"HIRE", "LAND"}:
            for offset in range(quantity):
                order = _purchase_order(key, 1)
                if order is None or slots <= 0:
                    break
                market.append(order)
                slots -= 1
                money -= _unit_purchase_cost(key, obs, offset)
            planned = quantity
        else:
            order = _purchase_order(key, quantity)
            if order is None:
                continue
            market.append(order)
            slots -= 1
            money -= sum(
                _unit_purchase_cost(key, obs, offset)
                for offset in range(quantity)
            )
            planned = quantity
        pending[key] = max(0, int(pending.get(key, 0)) - planned)

    action["market"] = market[:10]
    return action


def _town_demand_now(obs, item, step):
    demand = 1 if item != "FERTILIZER" and step % 24 == 0 else 0
    if step % 4 != 0:
        return demand
    town = _get(obs, "town", {}) or {}
    for shop in list(_get(town, "unlocked_shops", []) or []):
        products = _SHOP_PRODUCTS.get(shop, ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _future_quantity(step, item, branch=None):
    actions = _branch_data(branch)["actions"]
    future = step + 1
    if not 0 <= future < len(actions):
        return 0
    return sum(
        _order_quantity(order)
        for order in (actions[future].get("market") or [])
        if len(order) >= 3
        and order[0] == "SELL"
        and order[1] == item
    )


def _repay(action, state, step):
    if int(state.get("due_step", -1)) != step:
        return _copy_action(action)
    action = _copy_action(action)
    due = {
        str(item): max(0, int(quantity))
        for item, quantity in dict(state.get("due", {})).items()
    }
    market = []
    for raw_order in action.get("market") or []:
        order = list(raw_order)
        if (
            len(order) >= 3
            and order[0] == "SELL"
            and order[1] in due
            and due[order[1]] > 0
        ):
            quantity = _order_quantity(order)
            reduction = min(quantity, due[order[1]])
            quantity -= reduction
            due[order[1]] -= reduction
            if quantity <= 0:
                continue
            order[2] = quantity
        market.append(order)
    action["market"] = market[:10]
    state["due_step"], state["due"] = -1, {}
    return action


def _front_run(action, obs, state, step, branch=None):
    action = _copy_action(action)
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    reserves = _pickup_reserves(action)
    existing_sales = _existing_sales(action)
    moved = {}
    for item in _PREMIUM_ITEMS:
        target = (
            _future_quantity(step, item)
            if branch is None
            else _future_quantity(step, item, branch)
        )
        if target <= 0 or _town_demand_now(obs, item, step) > 0:
            continue
        stock = _inventory_count(shed, item)
        quantity = min(
            target,
            max(
                0,
                stock
                - int(reserves.get(item, 0))
                - int(existing_sales.get(item, 0)),
            ),
        )
        if quantity <= 0:
            continue
        market = [list(order) for order in action.get("market") or []]
        existing = next(
            (
                order
                for order in market
                if len(order) >= 3
                and order[0] == "SELL"
                and order[1] == item
            ),
            None,
        )
        if existing is not None:
            existing[2] = _order_quantity(existing) + quantity
        elif len(market) < 10:
            market.append(["SELL", item, quantity])
        else:
            continue
        action["market"] = market[:10]
        existing_sales[item] = existing_sales.get(item, 0) + quantity
        moved[item] = moved.get(item, 0) + quantity
    if moved:
        state["due_step"] = step + 1
        state["due"] = moved
    return action


def agent(obs):
    try:
        raw_step = int(_get(obs, "step", 0) or 0)
        step = min(max(0, raw_step), len(_ACTIONS) - 1)
        state = _reset_if_needed(obs, step)
        branch = _select_branch(obs, state, step)
        purchase = _purchase_state(obs, step)
        _reconcile_purchases(obs, purchase)
        action = _route_action(step, branch)
        if _ENABLE_FIELD_GUARDS:
            action = _weed_repair_action(obs, action, step, state, branch)
            action = _guard_field_actions(
                obs,
                action,
                step=step,
                weed_only=_weed_annotations(step, branch),
            )
        else:
            action = _align_hands(action, obs)
        if _ENABLE_PURCHASE_RECOVERY:
            action = _recover_purchases(
                action, obs, purchase, _ROUTE_TARGETS[branch][step]
            )
        if _ENABLE_SALE_CAP:
            action = _cap_sales(action, obs)
        front = state.setdefault("front", {"due_step": -1, "due": {}})
        if 0 <= int(front.get("due_step", -1)) < step:
            front["due_step"], front["due"] = -1, {}
        if _ENABLE_FRONT_RUN:
            action = _repay(action, front, step)
            action = _front_run(action, obs, front, step, branch)
        _record_purchase_attempts(action, obs, purchase)
        action["market"] = list(action.get("market") or [])[:10]
        return _align_hands(action, obs)
    except Exception:
        return {
            "farmer": ["PASS"],
            "hands": [
                ["PASS"] for _ in (_get(_farm(obs), "hands", []) or [])
            ],
            "market": [],
        }
