# Dynamic just intonation piano

2015-06-21

* Dynamicky dolaďovat MIDI tóny MIDI nástroje tak, aby výsledek byl co nejvíce harmonický.
* Problém: Přirozené ladění (podle vyšších harmonických) umožňuje hrát harmonicky v jedné tónině. Aby bylo možné hrát ve více tóninách, bylo by potřeba nástroj (např. klavír) přeladit. Aby nebylo potřeba nástroj neustále přelaďovat, je naladěn do kompromisního ladění, které má minimální chybu ve všech tóninách. Problém je, že v každé tónině má nějakou chybu, tj. nehraje tak harmonicky, jak je to možné.
* U virtuálních nástrojů nejsme omezeni laděním fyzických strun, ale můžeme tóny přelaďovat softwarem.
* Řešení: Program, který bude fungovat jako filtr mezi MIDI nástrojem a syntezátorem. Vstupem bude vzorkovaný zvuk ze syntezátoru a MIDI tóny z kláves. Program bude měřit disharmonii výstupního zvuku a podle toho dolaďovat jednotlivé aktuálně hrané tóny. Cílem je mít přesné frekvence tónů sice blízko základního ladění (např. equal temperament), ale přitom minimalizovat disharmonii.
* Na iOS by šlo možná využít AudioBusu.

V equal temeramentu jsou frekvence rovnoměrně distribuovány, aby měly hezké matematické vlastnosti. S jednou sadou frekvencí lze tak hrát v různých tóninách a celkem to zní. Lepší konsonanci lze zajistit přesným naladěním podle fyzikálních principů. Nevýhodou je přizpůsobení jedné tónině.

Obojí ale předpokládá statické ladění. Ve vokálních sborech ale lze ladit dynamicky tak, aby konsonance byle lepší než v ET. Omezení na jednu tóninu zde tedy není. Nevýhodou je ale, že se může sbor během písničky ujet kousek vedle.

Co zkusit toto naimplementovat pro virtuální nástroje, ovládané např. MIDI? Nebo dokonce jako filtr pro reálné nástroje, který by dělal pitch shifting pro každý tón zvlášť.

Principem by byla zpětná vazba mezi nastavením frekvence tónu a měřením konsonance ve výsledném zvuku.

Frekvence jednotlivých tónu by šlo jemně dolaďovat podle toho, co se zrovna hraje tak, aby byla konsonance co nejvyšší.

Dokonce by šlo případně i dolaďovat jednotlivé harmonické. Např. proto, aby se netloukly s harmonickými jiných tónů. Nebo proto, že některé nástroje nemají přesné harmonické. Nebo u některých nástrojů dokonce harmonické v čase lítají.

Kritéria pro optimalizaci:
\- konsonance výsledného zvuku
\- RMSE použitých frekvencí vůči equal. temperamentu.

Tj. výsledek by měl být více konsonantní, ale jako celek by neměl úplně ujíždět z původního ladění.

Algoritmus by se mohl učit - jako iniciální hodnoty použít výsledky předchozích optimalizací se stejnými vstupy.

<https://github.com/bzamecnik/ideas/blob/master/dynamic_tuning.md>
