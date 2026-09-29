# Error examples (test set, greedy decoding)

## Repeated words

- **EN:** i will instruct thee and teach thee in the way which thou shalt go: i will guide thee with mine eye.  
  **Ref:** ኣይኖቼን በአንተ ላይ አጠናለሁ።  
  **Seq2Seq-LSTM:** በምስክር መንገድ እወስዳለሁ፤ እኔም እወስዳለሁ፤ እኔም እመልስላለሁ፣ እመልስላለሁ።  
  **Attn-LSTM (Luong):** መንገድህን መንገድ መንገድህን አመሰግናለሁ፣ መንገድህንም እከተልሃለሁ።  
  **Attn-LSTM (Bahdanau):** በምሄድበት መንገድ እተማመናለሁ፣ በምስክሩም መንገድ አስተምራችኋለሁ።  

- **EN:** michael felt devastated when he realized how inconsiderate and unkind he had been.  
  **Ref:** ማይክል፣ የፈጸመው ድርጊት ምን ያህል አሳቢነትና ደግነት የጎደለው እንደሆነ ሲገነዘብ በጣም አዘነ።  
  **Seq2Seq-LSTM:** ሚስቴና የምናውቅ ነገር ምን ያህል እንደሆነ ለማወቅና ስለ እሱ ምን ያህል እንደሚታወቅ ያውቅ ነበር።  
  **Attn-LSTM (Luong):** ህዝቅያስ፣ ምን ያህል እንደተጨነቁና እንደተናገረችው ተሰምቶት ነበር።  
  **Attn-LSTM (Bahdanau):** ሚካኤል በንግግሩና በአእምሮው ላይ እንዴት እንደተስፋፋ ሲገነዘብ አልቀረም።  

- **EN:** and abraham set seven ewe lambs of the flock by themselves.  
  **Ref:** አብርሃምም ሰባት ቄቦች በጎችን ለብቻቸው አቆመ።  
  **Seq2Seq-LSTM:** አብርሃምም ሰባት በጎችን ሰባት አውራ በጎችን ሰባት አውራ በጎችን ላከ።  
  **Attn-LSTM (Luong):** አብርሃምም ከመንጋው ጋር ሰባት በሬዎችን አቀረበ።  
  **Attn-LSTM (Bahdanau):** አብርሃምም ከመንጋው መካከል ከመንጋው መካከል በጎችን ሰባት በጎችን አቀረበ።  

- **EN:** the lazy one says: "there is a young lion in the road,a lion in the public square!"  
  **Ref:** ሰነፍ "በመንገድ ላይ ደቦል አንበሳ፣በአደባባይም አንበሳ አለ!" ይላል።  
  **Seq2Seq-LSTM:** አንበሳው "በአንበሳ ላይ የሚጮኹ አንበሳ፣በአንበሳም ላይ የሚፈራ አንበሳ ነው!" ይላል።  
  **Attn-LSTM (Luong):** ንብረቷ " አንበሳ አንበሳ አለ፤ አንበሳ አንበሳም አንበሳ ነው!" ይልሻል።  
  **Attn-LSTM (Bahdanau):** ሰነፍ ሰው "በመንገድ ላይ ደቦል አንበሳ፣በበአማማ ሜዳ ላይ አንበሳ ነው!" ይላል።  

## Missing words (under-translation)

- **EN:** michael felt devastated when he realized how inconsiderate and unkind he had been.  
  **Ref:** ማይክል፣ የፈጸመው ድርጊት ምን ያህል አሳቢነትና ደግነት የጎደለው እንደሆነ ሲገነዘብ በጣም አዘነ።  
  **Seq2Seq-LSTM:** ሚስቴና የምናውቅ ነገር ምን ያህል እንደሆነ ለማወቅና ስለ እሱ ምን ያህል እንደሚታወቅ ያውቅ ነበር።  
  **Attn-LSTM (Luong):** ህዝቅያስ፣ ምን ያህል እንደተጨነቁና እንደተናገረችው ተሰምቶት ነበር።  
  **Attn-LSTM (Bahdanau):** ሚካኤል በንግግሩና በአእምሮው ላይ እንዴት እንደተስፋፋ ሲገነዘብ አልቀረም።  

- **EN:** the time limit for receiving comments shall normally be 60 days, unless otherwise agreed by the concerned states.  
  **Ref:** ሚመለከታቸው አገራት መካከል የተለየ ስምምነት ከሌለ በስተቀር የምርመራ ረቂቅ ሪፖርት ላይ አስተያየት የመቀበያ ጊዜ ፷ ቀን ብቻ ነው  
  **Seq2Seq-LSTM:** የጊዜው ክፍያ የሚጠበቅበት ጊዜ የሚጠበቅበት ጊዜ ብቻ ሳይሆን አይቀርም።  
  **Attn-LSTM (Luong):** ለምናሳየው ጊዜ እልባት የሚጠይቅበት ጊዜ ቢኖርም እንኳ ለምናሳየው ነገር ሁሉ መፍትሄ አይሰጥም ወይም አይፈቅድም ወይም አይን አይፈቅድም ማለት አይደለም።  
  **Attn-LSTM (Bahdanau):** ለምታምንበት ጊዜ የሚጠበቅበት ጊዜ ቢኖርም ከምንጊዜውም በላይ የሆነ አንድ ቀን፣ ጥፋቱ ከጥፋቱ ጋር የሚመሳሰልበት ጊዜ አለ።  

- **EN:** when she finds out, she is angry and calls him inconsiderate.  
  **Ref:** ሚስቱ ይህን ስታውቅ ስለ እሷ ምንም እንደማያስብ በቁጣ ትናገራለች።  
  **Seq2Seq-LSTM:** እሷም ስትወስድ፣ በትላልቅም ላይ ትቆጣለች።  
  **Attn-LSTM (Luong):** እሷም ስትወስድ በጭንቀት ትዋጣለች፤ ደግሞም ትሞታለች።  
  **Attn-LSTM (Bahdanau):** ስትወጣም ተደናቅፈው በትራብ ትሰራለች።  

- **EN:** cursed be he that taketh reward to slay an innocent person. and all the people shall say, amen.  
  **Ref:** የንጹሁን ሰው ነፍስ ለመግደል ጉቦ የሚቀበል ርጉም ይሁን፤ ህዝቡም ሁሉ አሜን ይላሉ።  
  **Seq2Seq-LSTM:** ለችግረኛው ሁሉ ይንገሩ።  
  **Attn-LSTM (Luong):** ለምስክሩ የሚሟገት ሰው ይገደል። ህዝቡም ሁሉ ይንከራተሉ።  
  **Attn-LSTM (Bahdanau):** ንጹህ ሰውን ይገድል ዘንድ ይከፍታሉ። ህዝቡም ሁሉ ይአሜን።  

## Additional words (over-translation)

- **EN:** i will instruct thee and teach thee in the way which thou shalt go: i will guide thee with mine eye.  
  **Ref:** ኣይኖቼን በአንተ ላይ አጠናለሁ።  
  **Seq2Seq-LSTM:** በምስክር መንገድ እወስዳለሁ፤ እኔም እወስዳለሁ፤ እኔም እመልስላለሁ፣ እመልስላለሁ።  
  **Attn-LSTM (Luong):** መንገድህን መንገድ መንገድህን አመሰግናለሁ፣ መንገድህንም እከተልሃለሁ።  
  **Attn-LSTM (Bahdanau):** በምሄድበት መንገድ እተማመናለሁ፣ በምስክሩም መንገድ አስተምራችኋለሁ።  

- **EN:** he has devastated my whole household.  
  **Ref:** መላ ቤተሰቤን አጥፍቷል።  
  **Seq2Seq-LSTM:** በመላው ምድር ላይ የምኖርበት ጊዜ አለ።  
  **Attn-LSTM (Luong):** መላውን ቤተሰብ ሁሉ ቤተሰቤን አጠናክሮታል።  
  **Attn-LSTM (Bahdanau):** መላውን ቤተሰብ ረስቷል።  

- **EN:** and abraham set seven ewe lambs of the flock by themselves.  
  **Ref:** አብርሃምም ሰባት ቄቦች በጎችን ለብቻቸው አቆመ።  
  **Seq2Seq-LSTM:** አብርሃምም ሰባት በጎችን ሰባት አውራ በጎችን ሰባት አውራ በጎችን ላከ።  
  **Attn-LSTM (Luong):** አብርሃምም ከመንጋው ጋር ሰባት በሬዎችን አቀረበ።  
  **Attn-LSTM (Bahdanau):** አብርሃምም ከመንጋው መካከል ከመንጋው መካከል በጎችን ሰባት በጎችን አቀረበ።  

## Named entities

- **EN:** thy seed will i establish for ever, and build up thy throne to all generations. selah.  
  **Ref:** ዘርህን ለዘላለም አዘጋጃለሁ፣ ዙፋንህንም ለልጅ ልጅ እመሰርታለሁ።  
  **Seq2Seq-LSTM:** ፍርዶችህ ለዘላለም ጸንቶ ይኖራል፤ ለዘላለምም ለዘላለም ጸንቶ ይኖራል።  
  **Attn-LSTM (Luong):** ዘርህ ለዘላለም፣ ዘርህንም ለዘላለም እጠብቃለሁ።  
  **Attn-LSTM (Bahdanau):** ዘርህ ለዘላለም ትጠብቃቸዋለች፣ ዙፋንህንም ከትውልድ እስከ ትውልድ ድረስ አጸናለሁ።  

- **EN:** religion: animist. christian and moslem  
  **Ref:** ሃይማኖት አረመኔ ክርስቲያንና እስላም  
  **Seq2Seq-LSTM:** ሃይማኖት እስላም አረመኔ  
  **Attn-LSTM (Luong):** ሃይማኖት እስላም አረመኔና አረመኔ  
  **Attn-LSTM (Bahdanau):** ሃይማኖት እስላም ክርስቲያንና ጥቂት  

- **EN:** this regulation shall come in to force as of its publication on megelete oromia.  
  **Ref:** ይህ ደንብ በመገለተ ኦሮሚያ ላይ ታትሞ ከወጣበት ቀን ጀምሮ ስራ ላይ የሚውል ይሆናል  
  **Seq2Seq-LSTM:** ይህ አዋጅ በመጪው የፌደራል መንግስት ላይ ተፈጻሚነት ይኖረዋል  
  **Attn-LSTM (Luong):** ይህ አዋጅ በትውልዶቹ ላይ በሚወጣው ደንብ የሚወሰን ይሆናል  
  **Attn-LSTM (Bahdanau):** ይህ ደንብ በቦርዱ ጅምስ ላይ በወጣው ጅምት ላይ በሚወጣው መመሪያ መሰረት ይሆናል  

- **EN:** sing praises to jehovah, (selah)  
  **Ref:** ለያህዌ የውዳሴ መዝሙር ዘምሩ፤ (ሴላ)  
  **Seq2Seq-LSTM:** ለያህዌ የውዳሴ መዝሙር ዘምሩ፤  
  **Attn-LSTM (Luong):** ለያህዌ የውዳሴ መዝሙር ዘምሩ ()  
  **Attn-LSTM (Bahdanau):** ለያህዌ የውዳሴ መዝሙር ዘምሩ (ሴላ)  

- **EN:** jon: no, i don't recall.  
  **Ref:** ኢዮብ፣ አይ፣ አላስታውስም።  
  **Seq2Seq-LSTM:** ኢዮብ፣ አዎ፣ አይቻለሁ።  
  **Attn-LSTM (Luong):** ኢዮብ፣ አዎ፣ ትክክል አይደለሁም።  
  **Attn-LSTM (Bahdanau):** ኢዮብ፣ አይ፣ አይናገርም።  

## Unknown / rare words

- **EN:** michael felt devastated when he realized how inconsiderate and unkind he had been.  
  **Ref:** ማይክል፣ የፈጸመው ድርጊት ምን ያህል አሳቢነትና ደግነት የጎደለው እንደሆነ ሲገነዘብ በጣም አዘነ።  
  **Seq2Seq-LSTM:** ሚስቴና የምናውቅ ነገር ምን ያህል እንደሆነ ለማወቅና ስለ እሱ ምን ያህል እንደሚታወቅ ያውቅ ነበር።  
  **Attn-LSTM (Luong):** ህዝቅያስ፣ ምን ያህል እንደተጨነቁና እንደተናገረችው ተሰምቶት ነበር።  
  **Attn-LSTM (Bahdanau):** ሚካኤል በንግግሩና በአእምሮው ላይ እንዴት እንደተስፋፋ ሲገነዘብ አልቀረም።  

- **EN:** these principles, plainly stated centuries ago in god's law to israel, can still be useful in courtrooms today.  
  **Ref:** ከብዙ መቶ ዘመናት በፊት ለእስራኤላውያን በተሰጠው የአምላክ ህግ ውስጥ በግልጽ የተቀመጡት እነዚህ መመሪያዎች በዛሬው ጊዜም ፍርድ ለመስጠት የሚረዱ ጠቃሚ መመሪያ ሆነው ሊያገለግሉ ይችላሉ።  
  **Seq2Seq-LSTM:** እነዚህ ሰዎች፣ እስራኤላውያን በምግብና በምስጢርታዊ ድርጊቶች ላይ የተመሰረተው አምላክ እንደሆነ የሚያረጋግጥ ማስረጃ እንደሆነ ይሰማቸዋል።  
  **Attn-LSTM (Luong):** እነዚህ መመሪያዎች፣ እስራኤላውያን በህብረት በመስራት ውስጥ የአምላክን ህግ በስራ ላይ የሚታዩት በዛሬው ጊዜ በአርማጌዶን ውስጥ ያሉ አንዳንድ መስዋእቶችን ሊቋቋም ይችላል።  
  **Attn-LSTM (Bahdanau):** እነዚህ መመሪያዎች በዛሬው ጊዜ ያሉ ህግ በአምላክ ህግ ውስጥ በዛሬው ዘመን በዛሬው ዘመን ውስጥ ጠቃሚ የሆኑ አንዳንድ ጠቃሚ ሃሳቦችን ይጠቀሙበታል።  

- **EN:** without fear of reprisals in such countries, citizens were free to discuss religious matters and to disagree openly with the established churches.  
  **Ref:** እንዲህ ባሉ አገሮች ውስጥ የሚኖሩ ሰዎች ቅጣት እንደሚደርስባቸው ሳይፈሩ በሃይማኖታዊ ርእሰ ጉዳዮች ላይ መወያየት እንዲሁም የአብያተ ክርስቲያናትን ትምህርት እንዳልተቀበሉ በይፋ መግለጽ ይችሉ ነበር።  
  **Seq2Seq-LSTM:** እነዚህ ሰዎች፣ ሃይማኖታዊ መሪዎች፣ ባህልና ባህል ያላቸው ሰዎችም ቢሆን በሃይማኖት ላይ የተመሰረተው አመጽ እንዲስፋፋ ለማድረግ ጥረት አድርገዋል።  
  **Attn-LSTM (Luong):** እንዲህ ያሉ አገሮችን የሚቃወሙት ሰዎች ከምን አንጻር ነጻ መውጣትና ከሃይል ጋር በተያያዘም ሁኔታው ተመሳሳይ እንደሆነ ግልጽ ነው።  
  **Attn-LSTM (Bahdanau):** አገሮች አገሮች ከምግብ ጋር በተያያዘም እንኳ ሃይማኖታዊ ጉዳዮችን በተመለከተ ተቃወሙ፤ እንዲሁም ከጉባኤው ጋር የሚስማማ እርምጃ ለመውሰድ ተስማማ።  

- **EN:** when she finds out, she is angry and calls him inconsiderate.  
  **Ref:** ሚስቱ ይህን ስታውቅ ስለ እሷ ምንም እንደማያስብ በቁጣ ትናገራለች።  
  **Seq2Seq-LSTM:** እሷም ስትወስድ፣ በትላልቅም ላይ ትቆጣለች።  
  **Attn-LSTM (Luong):** እሷም ስትወስድ በጭንቀት ትዋጣለች፤ ደግሞም ትሞታለች።  
  **Attn-LSTM (Bahdanau):** ስትወጣም ተደናቅፈው በትራብ ትሰራለች።  

## Numbers

- **EN:** the time limit for receiving comments shall normally be 60 days, unless otherwise agreed by the concerned states.  
  **Ref:** ሚመለከታቸው አገራት መካከል የተለየ ስምምነት ከሌለ በስተቀር የምርመራ ረቂቅ ሪፖርት ላይ አስተያየት የመቀበያ ጊዜ ፷ ቀን ብቻ ነው  
  **Seq2Seq-LSTM:** የጊዜው ክፍያ የሚጠበቅበት ጊዜ የሚጠበቅበት ጊዜ ብቻ ሳይሆን አይቀርም።  
  **Attn-LSTM (Luong):** ለምናሳየው ጊዜ እልባት የሚጠይቅበት ጊዜ ቢኖርም እንኳ ለምናሳየው ነገር ሁሉ መፍትሄ አይሰጥም ወይም አይፈቅድም ወይም አይን አይፈቅድም ማለት አይደለም።  
  **Attn-LSTM (Bahdanau):** ለምታምንበት ጊዜ የሚጠበቅበት ጊዜ ቢኖርም ከምንጊዜውም በላይ የሆነ አንድ ቀን፣ ጥፋቱ ከጥፋቱ ጋር የሚመሳሰልበት ጊዜ አለ።  

- **EN:** 16. why should we not settle for having a bible student read answers from a bible study aid?  
  **Ref:** 16. አንድ የመጽሃፍ ቅዱስ ተማሪ ከሚጠናው ጽሁፍ ላይ እያነበበ በሚሰጠው መልስ መርካት የሌለብን ለምንድን ነው?  
  **Seq2Seq-LSTM:** 16. መጽሃፍ ቅዱስን ለማጥናት የሚጠቅመን ለምን እንደሆነ መጽሃፍ ቅዱስ የሚሰጠው ለምንድን ነው?  
  **Attn-LSTM (Luong):** 16. መጽሃፍ ቅዱስን በማንበብ መጽሃፍ ቅዱሳዊ ጽሁፎች በማንበብ መጽሃፍ ቅዱሳዊ ጽሁፎች ማግኘት የምንችለው ለምንድን ነው?  
  **Attn-LSTM (Bahdanau):** 16. መጽሃፍ ቅዱስን ማጥናት እንድንችል መጽሃፍ ቅዱሳዊ ጽሁፎችን ማጠናችን የሌለብን ለምንድን ነው?  

- **EN:** (proverbs 18: 17) you will be more apt to apologize if you have a realistic view of yourself and your shortcomings.  
  **Ref:** (ምሳሌ 18፣ 17) ስለ ራስህና ስለ ድክመትህ ትክክለኛ አመለካከት መያዝህ ይቅርታ መጠየቅ ይበልጥ ቀላል እንዲሆንልህ ያደርጋል።  
  **Seq2Seq-LSTM:** (ምሳሌ 18፣ 17) አንተም የትዳር ጓደኛችሁን የምትወዳቸውን ነገሮችና ስሜታችሁን መቆጣጠር ትችላለህ።  
  **Attn-LSTM (Luong):** (ምሳሌ 18፣ 17) እንዲህ አይነት አመለካከት ቢያጋጥምህና ለምናሳየው ነገር ጥብቅ መሆን ትችላለህ።  
  **Attn-LSTM (Bahdanau):** (ምሳሌ 18፣ 17) አንተም ሆንክ ለህክምናና ለልጆቻችሁ አክብሮት ካለህ ይቅርታ መጠየቅ ትችላለህ።  

## Long sentences

- **EN:** that your error will cost you your lives. for you sent me to jehovah your god, saying, 'pray in our behalf to jehovah our god, and tell us everything that jehovah our god says, and we will do it.'  
  **Ref:** በደላችሁም ህይወታችሁን እንደሚያሳጣችሁ እወቁ። እንዲህ ስትሉ ወደ ይሆዋ አምላካችሁ ልካችሁኝ ነበርና: 'ወደ አምላካችን ወደ ይሆዋ ስለ እኛ ጸልይ፤ አምላካችን ይሆዋም የሚለውን ነገር ሁሉ ንገረን፤ እኛም የተባልነውን እናደርጋለን።'  
  **Seq2Seq-LSTM:** ለአባቴ እንዲህ ስትል ወደ ያህዌ ወደ አንተ መጣል 'ያህዌን ወደ አንተ ወደ አንተ ሂድ፤ ደግሞም እንዲህ ብለህ ጠይቅ፤ ደግሞም ወደ አንተ ይቀርባል' ይላል ይሆዋ።  
  **Attn-LSTM (Luong):** ይህም በህይወትህ ላይ በደል ያመጣህ። ይሆዋ ሆይ፣ እባክህ ስለ አምላካችንና ስለ አምላካችንን ሁሉ እንወስዳለን።  
  **Attn-LSTM (Bahdanau):** ይህ በደልህ በህይወትህ ላይ ይጣሉ። 'አምላኬን ወደ አምላካችን ወደ ይሆዋ እንጸልይና፣ አምላካችንም አምላካችንን ሁሉ እንመልከት፤ ደግሞም እኛም አምላካችንን እንመልከት' በማለት ወደ ይሆዋ አመጣሃል።  

- **EN:** they feel much as did peter when he said under inspiration: "praised be the god and father of our lord jesus christ, for according to his great mercy he gave us a new birth to a living hope through the resurrection of jesus christ from the dead, to an incorruptible and undefiled and unfading inheritance.  
  **Ref:** ጴጥሮስ በመንፈስ መሪነት የሚከተለውን ሃሳብ ሲያሰፍር የነበረው አይነት ስሜት አላቸው፣ "የጌታችን የኢየሱስ ክርስቶስ አምላክና አባት ይወደስ፤ እሱ በታላቅ ምህረቱ በኢየሱስ ክርስቶስ ትንሳኤ አማካኝነት ለህያው ተስፋ እንደ አዲስ ወልዶናልና፤ እንዲሁም ለማይበሰብስ፣ ለማይረክስና ለማይጠፋ ርስት ወልዶናል።  
  **Seq2Seq-LSTM:** ኢየሱስ ክርስቶስን እንዲህ ብሎ ነበር፣ "ክርስቶስ፣ አምላክና ክርስቶስን ከሞት አስነሳው፤ ምክንያቱም ኢየሱስ ክርስቶስን ከሞት አስነሳው፤ ምክንያቱም ኢየሱስ ክርስቶስን እንደወደደና ለክርስቶስ ፍቅር እንዳለን ማሳየት እንድንችል ከምንም በላይ ለክርስቶስ ፍቅር እንዳለን እናሳያለን።  
  **Attn-LSTM (Luong):** ጴጥሮስ፣ ኢየሱስ "የሰማይ አባት" እንደሆነና ኢየሱስ ክርስቶስን እንደ ምሳሌ የሚገልጹት እንደ ናዝራዊና ስለ እሱ ሲናገር ተስፋ ሰጥቶታል፤ ምክንያቱም በሰማይ የሚኖረውን ምድር ከሞት እንድንወጣ ሲል ለሞት የሚያበቃ ተስፋ ሰጥቶናል።  
  **Attn-LSTM (Bahdanau):** ጴጥሮስ በመንፈስ መሪነት እንዲህ ሲል ጸልዮአል: "በጌታችን ኢየሱስ ክርስቶስና የኢየሱስ ክርስቶስ ጌታ ኢየሱስ ክርስቶስን ከሞት በማስነሳት፣ ለክርስቶስ ምህረትና ለክርስቶስ ምህረት አሳልፎ ሰጥቶናል፤ እንዲሁም ከሞት በማስነሳት፣ ለወለድነውም ሆነ ለነፍሳችን ርስት አድርጎ ሰጥቷል።  

- **EN:** "'that is why the word of jehovah the god of israel is: "i did indeed say that your house and the house of your forefather would always walk before me." but now jehovah declares: "it is unthinkable, on my part, because those honoring me i will honor, but those despising me will be treated with contempt."  
  **Ref:** "'ስለዚህ የእስራኤል አምላክ የይሆዋ ቃል እንዲህ ይላል: "የአንተ ቤትና የአባትህ ቤት ምንጊዜም በፊቴ እንደሚሄዱ ተናግሬ ነበር።" አሁን ግን ይሆዋ እንዲህ ይላል: "ይህ ፈጽሞ የማላስበው ነገር ነው፤ ምክንያቱም የሚያከብሩኝን አከብራለሁ፤ የሚንቁኝ ግን ይናቃሉ።"  
  **Seq2Seq-LSTM:** "'የእስራኤልን የእስራኤል ቤት እንዲህ አልኩ: 'እኔ ይሆዋ እንዲህ ይላል: "እኔም እኔ ከአንተ ጋር ነኝ" ይላል ይሆዋ።  
  **Attn-LSTM (Luong):** "'የእስራኤል አምላክ ይሆዋ እንዲህ ይላልና: "የይሆዋ ቃል በፊትህ ጸንቶ ይቀመጥ ነበር፤ እኔም በፊትህ ይገረማሉ። እኔ ይሆዋ ሆይ፣ እኔን የሚፈሩ ሰዎች ክብርና ክብር የሚፈሩ ሰዎች ሁሉ ክብር ያገኛሉ።"  
  **Attn-LSTM (Bahdanau):** "'የእስራኤል አምላክ ይሆዋ ሆይ፣ እኔና የአባትህ ቤት ምንጊዜም በፊቴ ይሄዳል።" ይላል ይሆዋ ግን "አዎ፣ እኔ የሚከብዱኝ፣ እኔ የሚገዙት ክብር፣ እኔ ግን ክብር ይገባኛል፤ እኔ ግን ክብር ይገባሉ" በማለት ይናገራል።  
