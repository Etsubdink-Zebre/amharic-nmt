# Error examples (test set, greedy decoding)

## Repeated words

- **EN:** whiston is also known, especially to bible students, for his translation into english of the writings of first century jewish historian flavius josephus.  
  **Ref:** ዊስተን በመጽሃፍ ቅዱስ ተማሪዎች ዘንድ ይበልጥ የሚታወቀው በመጀመሪያው መቶ ዘመን የኖረውን አይሁዳዊ ታሪክ ጸሃፊ ይኸውም የፍላቪየስ ጆሴፈስን ጽሁፎች ወደ እንግሊዝኛ በመተርጎሙ ነው።  
  **Seq2Seq-LSTM:** በተጨማሪም የግሪካውያን ታሪክ ጸሀፊ የሆኑት የግሪክዮስን የመጽሃፍ ቅዱስ ትርጉሞች በስምንተኛው መቶ ዘመን የነበሩ በርካታ ምሁራንን ጨምሮ የአይሁዳውያንን ታሪክ አብራራ።  
  **Attention-LSTM:** ዊስተንም በተለይ የመጽሃፍ ቅዱስ ተማሪዎች በተለይም በተለይ በመጀመሪያው መቶ ዘመን የነበሩ ጥንታዊ የታሪክ ምሁር የሆኑት ጆሴፈስስ፣ የመጽሃፍ ቅዱስ ተማሪው ጆሴስ ጆሴፍስ ኦቭ ዘ ቨርዥንስ ኦቭ ቨርስቲየስንስ የተባለው መጽሃፍ ቅዱስ ነው።  

- **EN:** (acts 15: 6 11, 13, 14, 28, 29) likely, both jewish and gentile christians appreciated peter's fearlessness in presenting the facts.  
  **Ref:** (ስራ 15፣ 6 11, 13, 14, 28, 29) አይሁዳውያኑም ሆኑ ከአህዛብ ወገን የመጡ ክርስቲያኖች፣ ጴጥሮስ ማስረጃዎቹን ያለምንም ፍርሃት በማቅረቡ ተደስተው እንደሚሆን ጥርጥር የለውም።  
  **Seq2Seq-LSTM:** (ስራ 15፣ 13፤ እብራውያን 13፣ 13, 13, 28) እነዚህ ክርስቲያኖች፣ አይሁዳውያን፣ አይሁዳውያንና እህቶቻችንም ቢሆኑ የአይሁዳውያንን እምነት የሚያንጸባርቁት አይሁዳውያን ነበሩ።  
  **Attention-LSTM:** (ስራ 15፣ 6 11, 13, 14, 28, 29) አይሁዳውያንና ከአህዛብ ወገን የነበሩት አይሁዳውያን ጴጥሮስን እውነታውን በማስፋፋት ረገድ ፍርሃት እንዲያድርባቸው አድርጎታል።  

- **EN:** (acts 14: 17; 17: 26 28) we can become acquainted with our heavenly father and his loving purposes for us by studying his word, the bible.  
  **Ref:** (የሃዋርያት ስራ 14፣ 17፤ 17፣ 26 28) በተጨማሪም አምላክ ቃሉን መጽሃፍ ቅዱስን ሰጥቶናል፤ መጽሃፍ ቅዱስን በማጥናት በሰማይ ስላለው አባታችን እንዲሁም ለእኛ ስላለው አላማ ማወቅ እንችላለን።  
  **Seq2Seq-LSTM:** (ስራ 17፣ 28, 17፤ 17፣ 17) አፍቃሪ በሆነው አባታችን አማካኝነት በሰማይ የሚኖረው አባታችንን እንዲሁም ቃሉን የምንወድና ቃሉን የምንጠብቅበት መንገድ ነው።  
  **Attention-LSTM:** (ስራ 14፣ 17፤ 17፣ 26 28) በሰማይ የሚኖረው አባታችን፣ ቃሉን በማጥናትና ቃሉን በማጥናት እንዲሁም የእሱን ፍቅር እንድናጠናክርልን ያስችለናል።  

- **EN:** this book of divine wisdom can help you set sound priorities, make wise decisions, and endure any present trials with joy and hope.  
  **Ref:** መለኮታዊ ጥበብ ያለው ይህ መጽሃፍ፣ ቅድሚያ ልትሰጣቸው የሚገቡ ነገሮችን ለይተህ እንድታውቅ፣ ጥበብ የተንጸባረቀበት ውሳኔ እንድታደርግና የሚያጋጥምህን ማንኛውንም ፈተና በደስታና በተስፋ እንድትወጣ ይረዳሃል።  
  **Seq2Seq-LSTM:** ይህ መለኮታዊ ጥበብ፣ ጥበብና ጥበብ የሚንጸባረቅበት ውሳኔ ማድረግ፣ ጥበብ የሚንጸባረቅበት አካሄድና ጥበብ የተንጸባረቀበት ውሳኔ ማድረግህ ጥበብን እንድታዳብር ይረዳሃል።  
  **Attention-LSTM:** ይህ መለኮታዊ ጥበብ ያዘለ ምክር ጥሩ እንዲሆን ማድረግ፣ ጥበብ የሚንጸባረቅበት ውሳኔ እንድታደርግና ጥበብ የሚንጸባረቅበትና ፈተናዎችን በጽናት ለመወጣት የሚያስችልህን ነገር በጥበብ እንድትቋቋም ሊረዳህ ይችላል።  

## Missing words (under-translation)

- **EN:** ending the developing life would be like an abortion.  
  **Ref:** በዚህ መንገድ፣ በማደግ ላይ ያለው ጽንስ እንዲቋረጥ ያደርጋል፤ ይህም ውርጃ እንደመፈጸም ይቆጠራል።  
  **Seq2Seq-LSTM:** ህይወትን እንደ ህይወት ማሳደድን ይቀጥላል።  
  **Attention-LSTM:** በህይወት መተርጎም እንደምወድ ይቆጠር ነበር።  

- **EN:** how did we come to be?  
  **Ref:** የሰው ልጆች ወደ ህልውና የመጡት እንዴት ነው?  
  **Seq2Seq-LSTM:** ታዲያ ምን ማድረግ እንችላለን?  
  **Attention-LSTM:** ታዲያ እኛስ እንዴት ነን?  

- **EN:** on the other hand, the bible states: "no matter what we ask according to his will, he hears us."  
  **Ref:** በሌላ በኩል ደግሞ መጽሃፍ ቅዱስ "የምንጠይቀው ነገር ምንም ይሁን ምን ከፈቃዱ ጋር በሚስማማ ሁኔታ እስከለመንን ድረስ ይሰማናል" በማለት ይናገራል።  
  **Seq2Seq-LSTM:** በሌላ በኩል ደግሞ መጽሃፍ ቅዱስ "የምድርን ነገር ሁሉ ይመረምራል" ይላል።  
  **Attention-LSTM:** በሌላ በኩል ደግሞ መጽሃፍ ቅዱስ "በእርግጥ የምንጠይቀውን ነገር ሁሉ ይሰማናል" ይላል።  

- **EN:** what i discovered strengthened my faith in god.  
  **Ref:** ይህን ሳደርግ ያወቅሁት ነገር በአምላክ ላይ ያለኝን እምነት አጠናከረው።  
  **Seq2Seq-LSTM:** በአምላክ ላይ እምነት በማሳደር ረገድ እምነቴን አጠናክሮልኛል።  
  **Attention-LSTM:** በአምላክ ላይ ያለኝን እምነት አጠናክሮልኛል።  

## Additional words (over-translation)

- **EN:** (acts 1: 8) jesus had earlier prepared them for such an extensive assignment by drawing their attention to good qualities in foreigners.  
  **Ref:** (ስራ 1፣ 8) ደቀ መዛሙርቱ ይህን ተልእኮ ለመወጣት፣ ኩራትንና ጭፍን ጥላቻን ማስወገድ ነበረባቸው።  
  **Seq2Seq-LSTM:** (ስራ 1፣ 8) ኢየሱስ፣ ሴቶችን ለማወደስ ብቁ እንዲሆኑላቸው፣ ለስምንት ንብረታቸውን ለማንጸባረቅ ጥረት አድርጓል።  
  **Attention-LSTM:** (ስራ 1፣ 8) ኢየሱስ ቀደም ሲል በባእድ አገር ያሉ ሰዎች ጥሩ ዜጎችን በመልካም ስራ ላይ እንዲውል በማድረግ ይህን ተልእኮ ተጠቅመዋል።  

- **EN:** now as soon as the 1,000 years have ended, satan will be released from his prison,  
  **Ref:** ይህ 1,000 ኣመት እንዳበቃም ሰይጣን ከእስራቱ ይፈታል፤  
  **Seq2Seq-LSTM:** አሁን ከ 40 አመት በኋላ እስከ ሞት ድረስ እስከ ሞት የሚደርስበት ጊዜ አንስቶ እስከ ሞት ይደርስበታል፤  
  **Attention-LSTM:** አሁን 1,000 አመት ያህል ጊዜው ሰይጣን ከእስር ቤት ይወሰዳል፤  

- **EN:** opposition to translation?  
  **Ref:** እንዳይተረጎም የተደረገውን ጥረት  
  **Seq2Seq-LSTM:** ለኢዮጵያን ትርጉም ምን ትርጉም አለው?  
  **Attention-LSTM:** ለውትድርና ተቃውሞ ተቋቁመው ይሆን?  

## Named entities

- **EN:** religion: animist. christian and moslem  
  **Ref:** ሃይማኖት አረመኔ ክርስቲያንና እስላም  
  **Seq2Seq-LSTM:** ሃይማኖት፣ ሃይማኖትና ሃይማኖት  
  **Attention-LSTM:** ሃይማኖት ክርስቲያንና ሞሪም  

- **EN:** and the people journeyed from kibrothhattaavah unto hazeroth; and abode at hazeroth.  
  **Ref:** ህዝቡም ከምኞት መቃብር ወደ ሀጼሮት ተጓዙ በሀጼሮትም ተቀመጡ።  
  **Seq2Seq-LSTM:** ህዝቡም ወደ ጊልያድ ወደ ናቡር ተራራ ተወሰደ፤ ወደ ንጋትም ሸሸ፤  
  **Attention-LSTM:** ህዝቡም ከቂብሮትሃትሃላታ ተነስተው ወደ ሃጼሮት ሄዱ።  

- **EN:** religion: animist and christian. defense:although nigeria has been under british protection, it maintains its own military organizations.  
  **Ref:** ሃይማኖት ብዙ እስላም ክርስቲያንና አረመኔ ምንም እንኳ እስከ ቅርብ ጊዜ በእንግሊዝ ስር ስትተዳደር ብትቆይም ናይጄሪያ የራሷ ወታደራዊ ድርጅት አላት  
  **Seq2Seq-LSTM:** ሃይማኖት፣ ሃይማኖትና ፖለቲካ፣ የኤርትራን የፖርቲን ክልላዊነትና የድርጅቱ አስተዳደርን በተመለከተ የጋራው ፕሬስን መቃወም  
  **Attention-LSTM:** ሃይማኖት፣ አቶና አቶ መለስ ዜና፣ የእንግሊዝ ማእከላዊና የእንግሊዝ ድርጅት አባል ሆኖ የታየውን የድርጅቱን ድርጅት ጠብቆ መኖር  

- **EN:** however, the belgian accent is quite different, so initially we had to overcome a language barrier.  
  **Ref:** ይሁንና የአነጋገር ቅላጼያቸው ለየት ያለ ነው፤ በመሆኑም መጀመሪያ ላይ ይህን ችግር መቋቋም ነበረብን።  
  **Seq2Seq-LSTM:** ይሁን እንጂ የፖርት ቋንቋ ተናጋሪዎች የቋንቋው ቋንቋ ተናጋሪው የቋንቋውን ቋንቋ ተጠቅመን ነበር።  
  **Attention-LSTM:** ይሁን እንጂ የፓርጂያን ቋንቋ ልዩ ቋንቋ ቢሆንም በጣም ተናግሮ የነበረ ቢሆንም መጀመሪያ ላይ ቋንቋውን ማሸነፍ አስፈልጎን ነበር።  

- **EN:** thy seed will i establish for ever, and build up thy throne to all generations. selah.  
  **Ref:** ዘርህን ለዘላለም አዘጋጃለሁ፣ ዙፋንህንም ለልጅ ልጅ እመሰርታለሁ።  
  **Seq2Seq-LSTM:** የመረጥከውን ዘር ሁሉ እባርክሃለሁ፤ ዙፋንህንም ለዘላለም አጸናለሁ።  
  **Attention-LSTM:** ዘርህ ለዘላለም ጸንቶ ይኖራል፤ ዙፋንህንም እስከ ትውልድ ድረስ አጸናለሁ።  

## Unknown / rare words

- **EN:** richard levins, a distinguished lawyer, told emlyn that he would be run down "like a wolf, without law or game."  
  **Ref:** የታወቀ ጠበቃ የነበረው ሪቻርድ ሊቨንስ ለኤምለን "ያለምንም ህግ ወይም ደንብ እንደ ተኩላ" ታድኖ እርምጃ እንደሚወሰድበት ነገረው።  
  **Seq2Seq-LSTM:** የፓርተርንያን የህብረት ዳይሬክተር "የምድርን ወይም የህጻንነትን መንጎቹን ወይም በህብረት ላይ የሚውል ሰው" እንደሆነ ተደርጎ ተገልጿል።  
  **Attention-LSTM:** የፖርቱጋል ኮሌክትሪክ ኮምፒውተር፣ "አንድ ሰው፣ ያለምንም ሰው ወይም ጨዋታ ሳይወርድ" እንደሚወርድ ገልጾ ነበር።  

- **EN:** to have authority over his princes as he pleasedand to teach his elders wisdom.  
  **Ref:** ይህም ደስ ባሰኘው መንገድ በመኳንንቱ ላይ እንዲሰለጥን፣ሽማግሌዎቹንም ጥበብ እንዲያስተምር ነው።  
  **Seq2Seq-LSTM:** በገዢው ላይ ስልጣንና ጥበብ የተሞላበት እርምጃ ይወስዳል፤  
  **Attention-LSTM:** መኳንንትን ደስ ያሰኛሉ፤ሽማግሌዎቹንም ለማስተማር አስተምሯቸዋል።  

- **EN:** willie and liz sneddon  
  **Ref:** ዊሊና ሊዝ ስኔደን  
  **Seq2Seq-LSTM:** ናኮር እና ንዴት  
  **Attention-LSTM:** ማሪ እና አጽፍን  

- **EN:** and israel journeyed, and spread his tent beyond the tower of edar.  
  **Ref:** እስራኤልም ከዚያ ተነሳ፣ ድንኳኑንም ከጋዴር ግንብ በስተ ወዲያ ተከለ።  
  **Seq2Seq-LSTM:** እስራኤልም ወደ ድንኳኑ ወስደው፤ ከዚያም ሌዋውያኑን ሰራ።  
  **Attention-LSTM:** እስራኤልም ጉዞ ተጎናጸፈው፤ ድንኳኑንም ከሴር ግንብ ይበልጥ አቀና።  

## Numbers

- **EN:** (prov. 15: 22) such spiritual people may tell you that the full time ministry provides an education that benefits you throughout life.  
  **Ref:** (ምሳሌ 15፣ 22) እንዲህ ያሉ መንፈሳዊ ሰዎች፣ የሙሉ ጊዜ አገልግሎት በህይወትህ ሙሉ የሚጠቅምህ ትምህርት እንደሚሰጥህ ይነግሩሃል።  
  **Seq2Seq-LSTM:** (ምሳሌ 15፣ 22) እንዲህ አይነት ህይወትህ ምንጊዜም ቢሆን መንፈሳዊ እድገት እንዲያደርጉልህ መርዳት ትችላለህ።  
  **Attention-LSTM:** (ምሳሌ 15፣ 22) እንዲህ ያሉ መንፈሳዊ ሰዎች በሙሉ ጊዜ አገልግሎትህ የሚጠቅም ትምህርት እንድታገኝ ሊገፋፉህ ይችላል።  

- **EN:** so all the days of kenan amounted to 910 years, and then he died.  
  **Ref:** ስለዚህ ቃይናን በአጠቃላይ 910 አመት ኖረ፤ ከዚያም ሞተ።  
  **Seq2Seq-LSTM:** በመሆኑም የ 20 አመት ልጅ ኖረ፤ ከዚያም ሞተ።  
  **Attention-LSTM:** በመሆኑም የሳንባ ዘመን ሁሉ 910 ኣመት ኖረ፤ ከዚያም ሞተ።  

- **EN:** (acts 15: 6 11, 13, 14, 28, 29) likely, both jewish and gentile christians appreciated peter's fearlessness in presenting the facts.  
  **Ref:** (ስራ 15፣ 6 11, 13, 14, 28, 29) አይሁዳውያኑም ሆኑ ከአህዛብ ወገን የመጡ ክርስቲያኖች፣ ጴጥሮስ ማስረጃዎቹን ያለምንም ፍርሃት በማቅረቡ ተደስተው እንደሚሆን ጥርጥር የለውም።  
  **Seq2Seq-LSTM:** (ስራ 15፣ 13፤ እብራውያን 13፣ 13, 13, 28) እነዚህ ክርስቲያኖች፣ አይሁዳውያን፣ አይሁዳውያንና እህቶቻችንም ቢሆኑ የአይሁዳውያንን እምነት የሚያንጸባርቁት አይሁዳውያን ነበሩ።  
  **Attention-LSTM:** (ስራ 15፣ 6 11, 13, 14, 28, 29) አይሁዳውያንና ከአህዛብ ወገን የነበሩት አይሁዳውያን ጴጥሮስን እውነታውን በማስፋፋት ረገድ ፍርሃት እንዲያድርባቸው አድርጎታል።  

## Long sentences

- **EN:** he honored jesus in an unexpected way by resurrecting him to "a superior position" and giving him what no one else had received up until that time immortal spirit life! (phil. 2: 9; 1 tim. 6: 16) what an outstanding acknowledgment of jesus' faithful course!  
  **Ref:** ኢየሱስን ፈጽሞ ባልተጠበቀ መንገድ አክብሮታል፤ ከሞት ካስነሳው በኋላ "የላቀ ቦታ የሰጠው" ከመሆኑም ሌላ እስከዚያ ጊዜ ድረስ ለማንም ተሰጥቶ የማያውቅ የማይሞት መንፈሳዊ ህይወት እንዲያገኝ አድርጓል! (ፊልጵ 2፣ 9፤ 1 ጢሞ 6፣ 16) በእርግጥም ኢየሱስ ለተከተለው የታማኝነት ጎዳና አስደናቂ በሆነ መንገድ እውቅና አግኝቷል!  
  **Seq2Seq-LSTM:** ኢየሱስ "የምድርን ህይወት" በማለት ጠርቶታል፤ ኢየሱስም "የአባቱ ህይወት" ሲል ጠርቶታል!  
  **Attention-LSTM:** ኢየሱስ "ለዘላለም ቦታ" ሲል ከሞት በኋላ ምንም አይነት ነገር ሳይኖረው ኢየሱስን ከሞት በማስነሳት ረገድ ኢየሱስን በመደገፍ ምንኛ አስደናቂ ነው! (ኤ 2፣ 9፤ 1 ጢሞ 6፣ 16) ኢየሱስ በታማኝነት ያከናወነውን የታማኝነት ጎዳና እንዴት ያለ ግሩም ምሳሌ ነው!  

- **EN:** august 31. mr. hammarkskjoeld compilaind that all belgian troops. had not been withdrawn. thirteen african states at leopeldville conference endorsed the united nations' work in the congo and called on mr. lumumba's government to co-operate with united nations  
  **Ref:** ነሀሴ ፳፭ ቀን ፲፱፻፶፪ ኣ.ም ሚስተር ሀመርሾልድ የቤልጁክ ወታደሮች በሙሉ ፈጽሞ ስላለመውጣታቸው ያላቸውን ቅሬታ አስታወቁ አስራ ሶስት የአፍሪካ ነጻ መንግስታታ በሊዎፖልድቪል ጉባኤያቸው የተባበሩት መንግስታት በኮንጎ ውስጥ የሚፈጽመውን ስራ በመደገፍ የሚስተር ሉሙምባ መንግስት ከተባበሩት መንግስታት ጋር እንዲተባበር ጠየቁ  
  **Seq2Seq-LSTM:** በአሜሪካ ፕሬዚዳንት የሻእቢያ መንግስት የሻእቢያን መንግስት ንብረቱና የሻእቢያ የሻእቢያ አምባገነን መንግስትን ያቀፈው የሻእቢያ ጦር ድርጅትን ንቀት ፕሬዚዳንት ማእከላዊ ሚኒስትር ማእከላዊ ማእከላዊ ፕሬዚዳንት ፕሬዝክትዌርን ፕሬልስጤት ንስር ፕሬስፒታል  
  **Attention-LSTM:** ነሀሴ 31 ቀን ጊልድ ጊልያድዋርድዋርስበርግ የተባበሩት መንግስታት ድርጅት በተባበሩት መንግስታት ላይ የተባበሩት መንግስታትን ስራ በበላይነት አቋቋመ።  

- **EN:** they stood before moses, eleazar the priest, the chieftains, and all the assembly at the entrance of the tent of meeting and said: "our father died in the wilderness, but he was not among the group who banded together against jehovah, the supporters of korah, but he died for his own sin and he did not have any sons.  
  **Ref:** እነሱም በመገናኛ ድንኳኑ መግቢያ በሙሴ፣ በካህኑ በአልአዛር፣ በአለቆቹ እንዲሁም በመላው ማህበረሰብ ፊት ቆመው እንዲህ አሉ፣ "አባታችን በምድረ በዳ ሞተ፤ ይሁንና እሱ በያህዌ ላይ ለማመጽ ከተባበሩት ከቆሬ ግብረ አበሮች አንዱ አልነበረም፤ እሱ የሞተው በራሱ ሃጢአት ነው፤ ወንዶች ልጆችም አልነበሩትም።  
  **Seq2Seq-LSTM:** እነሱም ሙሴ ከእስራኤላውያን መካከል አንዱን ተዉ፤ እሱም "ከእንግዲህ ጀምሮ በድንኳኑ ደጃፍ ላይ ተቀመጠ፤ ሆኖም ያህዌ በድንኳኑ ላይ የተቀመጠውን ሰው በድንኳኑ ውስጥ ተቀመጠ።  
  **Attention-LSTM:** እነሱም ሙሴን በካህኑ በመገናኛ ድንኳኑ መግቢያ ላይ በመገኘት በመገናኛ ድንኳኑ ደጃፍ ላይ ቀርበው እንዲህ አላቸው: "አባቴ በምድረ በዳ ሞተ፤ ሆኖም በሮም ላይ የተቀመጠው በምእራብ ላይ ነበር፤ ሆኖም የሬዛን ደጋፊዎች በሙሉ በያህዌ ፊት ባረከላቸው፤ እሱ ግን የገዛ ልጆቹን አልመለሰም፤ ሆኖም ወንዶች ልጆቹን አላወቀም።  
