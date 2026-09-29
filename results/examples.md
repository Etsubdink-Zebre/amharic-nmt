# Translation examples (test set, greedy decoding)

**1.** Source: and to it all the nations will stream.  
Reference: ብሄራትም ሁሉ ወደዚያ ይጎርፋሉ።  
Seq2Seq-LSTM: ብሄራትም ሁሉ ይንቀጠቀጣሉ።  
Attn-LSTM (Luong): በብሄራትም ሁሉ ላይ ይወድቃሉ።  
Attn-LSTM (Bahdanau): ብሄራትም ሁሉ ይጎናጸፋሉ።  

**2.** Source: let us see how this is so.  
Reference: እንዲህ የምንለው ለምን እንደሆነ እስቲ እንመልከት።  
Seq2Seq-LSTM: እስቲ ይህን ማድረግ የምንችለው እንዴት እንደሆነ እስቲ እንመልከት።  
Attn-LSTM (Luong): እስቲ ይህን እንዴት እንደ ምሳሌ እንመልከት።  
Attn-LSTM (Bahdanau): ይህ እንዴት እንደሆነ እስቲ እንመልከት።  

**3.** Source: for his loyal love toward us is great;  
Reference: ለእኛ ያሳየው ታማኝ ፍቅር ታላቅ ነውና፤  
Seq2Seq-LSTM: ታማኝ ፍቅሩ ለዘላለም ነውና፤  
Attn-LSTM (Luong): ታማኝ ፍቅሩ ታላቅ ነው፤  
Attn-LSTM (Bahdanau): ለእኛ ታላቅ ፍቅሩ ለእኛ ታላቅ ነውና፤  

**4.** Source: fear and trembling come upon me,  
Reference: ፍርሃት አደረብኝ፤ ደግሞም ተንቀጠቀጥኩ፤  
Seq2Seq-LSTM: ፍርሃትና ፍርሃት ተንቀጠቀጡ፤  
Attn-LSTM (Luong): ፍርሃትና ፍርሃት ተፈራኝ፤  
Attn-LSTM (Bahdanau): ፍርሃትና በፍርሃት ተናወጠ፤  

**5.** Source: spanish is the official language of honduras.  
Reference: የሆንዱራስ የስራ ቋንቋ ስፓንኛ ነው።  
Seq2Seq-LSTM: የእንግሊዝኛ ቋንቋ የእንግሊዝኛ ቋንቋ ነው።  
Attn-LSTM (Luong): የፓርጂ ቋንቋ ቋንቋ የቋንቋው ቋንቋ ነው።  
Attn-LSTM (Bahdanau): የእንግሊዝኛ ቋንቋ የቋንቋው ቋንቋ ነው።  

**6.** Source: praise jehovah for his great works  
Reference: ለታላላቅ ስራዎቹ ያህዌን አወድሱ  
Seq2Seq-LSTM: ለታላቅ ውዳሴ ምስጋና አቅርቡ  
Attn-LSTM (Luong): ያህዌን አወድሱ፤  
Attn-LSTM (Bahdanau): ያህዌ ታላቅ ስራው ለወደቁ  

**7.** Source: unlike other officials, he nerves feels like neither receiving nor giving gifts.  
Reference: እንደሌሎቹ አለቆችም ገጸ በረከት መቀበልም ሆነ መስጠት አይቃጣቸውም።  
Seq2Seq-LSTM: ሌሎች ደግሞ ሌሎችም ሆነ ሌሎች ሰዎችም እንኳ ለምግብና ለምግብ ሲሉ ለምግብ ሲሉም ተመሳሳይ ነገር አድርገዋል።  
Attn-LSTM (Luong): እንደ ሌሎች ባለስልጣናት፣ እንደ ንብረታቸውና እንደ መብረሩ የሚያገለግሉት ሰዎች ስጦታ አይሰጥም።  
Attn-LSTM (Bahdanau): ከሌሎች ሌሎች ባለስልጣናት ጋር በተያያዘም እንኳ አክብሮት እንደሌለው ስጦታ ለመስጠት ፈቃደኛ መሆን አለበት።  

**8.** Source: left: the convent in zaragoza, spain; right: nácar colunga bible translation  
Reference: በዛራጎዛ፣ ስፔን የሚገኘው ገዳም (በስተ ግራ) የናካር ኮሉንጋ የመጽሃፍ ቅዱስ ትርጉም (በስተ ቀኝ)  
Seq2Seq-LSTM: ዋና ከተማ፣ እንግሊዝኛ፣ እንግሊዝኛ፣ እንግሊዝኛ፣ እንግሊዝኛ፣ እንግሊዝኛ፣ እንግሊዝኛ፣ እንግሊዝኛ፣ እንግሊዝኛ  
Attn-LSTM (Luong): የስብሰባው፣ በ 1943፣ በ 1943፣ በ 1943 የስፔን፣ የስፔን ከተማ  
Attn-LSTM (Bahdanau): ግራ፣ በኦሮብ፣ በፔንዳጊል፣ በኦሮሚያ በምትገኘው በፓርጂያ ቋንቋ የተዘጋጀው የመጽሃፍ ቅዱስ ትርጉም  

**9.** Source: "i am making you a fortified copper wall to this people.  
Reference: "ለዚህ ህዝብ ጠንካራ የመዳብ ቅጥር አደርግሃለሁ።  
Seq2Seq-LSTM: "የተከረከረ ገመድ፣  
Attn-LSTM (Luong): "በምኩራብ ቅጥር ላይ የምጨምጥ ህዝብ ናችሁ።  
Attn-LSTM (Bahdanau): "ይህ ህዝብ የምፈርዱበት የመዳብ ቅጥር አደርግሃለሁ።  

**10.** Source: soon this kingdom will cause god's will to be done everywhere on earth.  
Reference: ይህ መንግስት በቅርቡ የአምላክ ፈቃድ በመላው ምድር ላይ እንዲፈጸም ያደርጋል።  
Seq2Seq-LSTM: ይህ መንግስት በምድር ላይ የአምላክ መንግስት በምድር ላይ እንዲኖሩ ያደርጋል።  
Attn-LSTM (Luong): በቅርቡ በቅርቡ የአምላክ መንግስት በምድር ላይ እንዲፈጸም ያደርጋል።  
Attn-LSTM (Bahdanau): ብዙም ሳይቆይ ይህ መንግስት የአምላክን ፈቃድ በምድር ላይ እንዲፈርስ ያደርጋል።  

**11.** Source: the bible uses the word "fornication" for some forms of sexual activity outside marriage.  
Reference: መጽሃፍ ቅዱስ ከጋብቻ ውጪ የሚፈጸሙ አንዳንድ ወሲባዊ ድርጊቶችን ለመግለጽ "ዝሙት" የሚለውን ቃል ይጠቀማል።  
Seq2Seq-LSTM: መጽሃፍ ቅዱስ "የተወሰነ ሰው" የሚለው አገላለጽ፣ የጾታ ብልግናን የሚያመለክተው የጾታ ብልግናን ማስወገድ ነው።  
Attn-LSTM (Luong): መጽሃፍ ቅዱስ "የተወሰነች" የጾታ ብልግናን የሚያመለክት የጾታ ብልግናን ለማመልከት ይገልጻል።  
Attn-LSTM (Bahdanau): መጽሃፍ ቅዱስ "ለመግባት" የሚለው ቃል በትዳር ውስጥ የጾታ ግንኙነት መፈጸም ተገቢ ነው።  

**12.** Source: and moses and eleazar the priest spake with them in the plains of moab by jordan near jericho, saying,  
Reference: ሙሴና ካህኑ አልኣዛር በዮርዳኖስ አጠገብ በኢያሪኮ ፊት ለፊት በሞኣብ ሜዳ ላይ።  
Seq2Seq-LSTM: ሙሴና ካህኑ በዮርዳኖስ ማዶ በዮርዳኖስ አጠገብ በዮርዳኖስ አጠገብ ባለው በዮርዳኖስ አጠገብ እንዲህ ብሎ ተናገረ:  
Attn-LSTM (Luong): ሙሴና ካህኑ በዮርዳኖስ አጠገብ በዮርዳኖስ አጠገብ በዮርዳኖስ አጠገብ በዮርዳኖስ አጠገብ እንዲህ ብሎ ነገራቸው:  
Attn-LSTM (Bahdanau): ሙሴምና ካህኑ አልኣዛር እንዲህ ብሎ በኢያሪኮ በኢያሪኮ አጠገብ በሞኣብ ሜዳ ላይ እንዲህ ብለው ተናገሩ።  

**13.** Source: children around the world believe that santa claus or father frost will bring them good luck, and presents, for the holidays.  
Reference: በኣለም ያሉ ህጻናት ሳንታ ክላውስ ወይም ፋዘር ፍሮስት ለበኣላት ጥሩ እድል፣ ስጦታ ይዘው እንደሚመጡላቸው ያምናሉ።  
Seq2Seq-LSTM: ልጆችን ጨምሮ በኢንተርኔት ላይ የሚገኙ ሰዎች፣ የምናገኛቸውን ነገሮች በሙሉና ለልጆችና ለምግብ ሲሉ ለምግብ ሲሉ ለምግብ ሲሉ ለምግብ ይዳርጋቸዋል።  
Attn-LSTM (Luong): በቤቴል የሚኖሩ ሰዎች፣ በምድራችንና በምስክሩ ላይ የሚታዩት ልጆችም ሆነ የበኩላቸውን ነገር ሁሉ ያድራሉ።  
Attn-LSTM (Bahdanau): በአለም ዙሪያ የሚገኙ ልጆች በምሬትም ሆነ በአባቴ ላይ ያሉ ልጆች ያሏቸውን መልካም ነገሮች፣ የበጎችና የበለጸጉትን ዘሮች ያወራሉ።  

**14.** Source: but if the blotch on his skin is white and its appearance is not deeper than the skin and the hair has not turned white, the priest will then quarantine the infected person for seven days.  
Reference: ሆኖም በቆዳው ላይ ያለው ቋቁቻ ነጭ ከሆነና ከቆዳው ዘልቆ የገባ ካልሆነ እንዲሁም በዚያ ቦታ ላይ ያለው ጸጉር ወደ ነጭነት ካልተለወጠ ካህኑ ቁስል የወጣበትን ሰው ለሰባት ቀን ተገልሎ እንዲቆይ ያደርገዋል።  
Seq2Seq-LSTM: ሆኖም ቁስሉ ከውስጥና ከውስጥና ከውስጥ ውጭ ሆኖ ቢወጣም እንኳ ቁስሉን ከውስጥና ከወይን ጠጅ ላይ ቢገኝ ካህኑ ቁስሉን ይነካል።  
Attn-LSTM (Luong): ሆኖም ቁስሉ ከምግብ ቆዳ ቆዳው ላይ ከቀጠለ ወይም ቁስሉን ከነካው ቁስሉ ቁስሉን ቢያጠፋ ቁስሉ ቁስሉን ለሰባት ቀን አይወስድም።  
Attn-LSTM (Bahdanau): ሆኖም ቆዳው ነጭ ከሆነ ነጭነቱ ቆዳውና ቆዳው ነጭ ከሆነ ነጭ ልብስ አይጨምርም፤ ካህኑ ቁስሉን ለሰባት ቀን ካህኑን ያያል።  

**15.** Source: consider: researchers speculate that some types of birds use the earth's magnetic field for navigation, as if they had a compass built into their brain.  
Reference: እስቲ የሚከተለውን አስብ፣ አንዳንድ አእዋፍ በአንጎላቸው ውስጥ የተገጠመ ኮምፓስ ያላቸው ያህል አቅጣጫን የሚያውቁት በምድር መግነጢሳዊ መስክ (ማግኔቲክ ፊልድ) አማካኝነት እንደሆነ ተመራማሪዎች ይገምታሉ።  
Seq2Seq-LSTM: እስቲ የሚከተለውን አስብ፣ ተመራማሪዎችን፣ የምእራፍ ዝርያ፣ ወፎች፣ የምእራብ ምንጭ፣ የምእራብ ምንጭ፣ የምእራብ ምንጭ እንደሆኑ ይሰማቸዋል።  
Attn-LSTM (Luong): እስቲ የሚከተለውን አስብ፣ ወፎች ወፎችን እንደ ምሳሌ ተጠቅመው ወፎችን እንደ ምሳሌ ተጠቅመው ወፎችን እንደ ምሳሌ ተጠቅመው እንደ እንስሳት ያሉ ወፎችን እንደ ምሳሌ እንውሰድ።  
Attn-LSTM (Bahdanau): እስቲ የሚከተለውን አስብ፣ ተመራማሪዎች ወፎች፣ ወፎች፣ በምስክሩ ላይ የተገነባው የሸክላ እቃ እንደ ምሳሌ መሸፈን እንዲችሉ የታገደባቸው አንዳንድ እጸዋትን እንደ ምሳሌ እንመርምር።  

**16.** Source: all the israelites from 20 years old and up who could serve in the army in israel were registered by their paternal house, and the total number registered was 603, 550.  
Reference: በእስራኤል ውስጥ ወደ ሰራዊቱ መቀላቀል የሚችሉት እድሜያቸው 20 አመትና ከዚያ በላይ የሆኑት እስራኤላውያን ሁሉ በየአባቶቻቸው ቤት ተመዘገቡ፤ የተመዘገቡትም ሰዎች ጠቅላላ ቁጥር 603,550 ነበር።  
Seq2Seq-LSTM: እስራኤላውያን በሙሉ ከተመዘገቡት 20,400 ነበሩ፤ በጊልጋልም የተመዘገቡት በአጠቃላይ 20,400 ነበሩ፤ በአጠቃላይ 2050 ነበሩ።  
Attn-LSTM (Luong): እስራኤላውያን በሙሉ በየአባቶቻቸው ቤት የተመዘገቡት እድሜያቸው 20,400 ነበሩ። እነሱም በየአባቶቻቸው ቤት የተመዘገቡትና የተመዘገቡት በአጠቃላይ 60,400 ነበሩ።  
Attn-LSTM (Bahdanau): በእስራኤል ውስጥ እድሜያቸው 20 ኣመትና ከዚያ በላይ የሆኑት እስራኤላውያን በሙሉ በየአባቶቻቸው ቤት በዝርዝር የተመዘገቡት ወንዶች ሁሉ በየአባቶቻቸው ቤት በዝርዝር የተመዘገቡት በአጠቃላይ 2,3,300 ነበሩ።  
