import matplotlib.pyplot as plt

def oblicz_portfel(roczny_zwrot, wklad_miesieczny, ilosc_lat, roczne_wydatki, inflacja=0.03, miesiace_bez_wplatow=None):
    okres_miesieczny = ilosc_lat * 12
    miesieczny_zwrot = (1 + roczny_zwrot) ** (1 / 12) - 1
    miesieczna_inflacja = (1 + inflacja) ** (1 / 12) - 1

    wartosc_portfela = [0]
    wklad_wlasny = [0]
    saldo = 0
    zysk_roczny = []
    zysk_roczny_procentowy = []
    procentowy_wzrost_portfela = []
    fire_miesiac = None

    docelowa_wartosc_fire = 25 * roczne_wydatki
    docelowa_wartosc_fire_realna = docelowa_wartosc_fire

    for miesiac in range(1, okres_miesieczny + 1):
        # dodawanie wkładu (z uwzględnieniem miesiąc_bez_wplatow)
        if miesiace_bez_wplatow is None or miesiac <= miesiace_bez_wplatow:
            saldo += wklad_miesieczny
        # kapitalizacja
        saldo *= (1 + miesieczny_zwrot)
        wartosc_portfela.append(saldo)

        # kumulacja wkładu własnego
        if miesiace_bez_wplatow is None or miesiac <= miesiace_bez_wplatow:
            wklad_wlasny.append(wklad_miesieczny * miesiac)
        else:
            wklad_wlasny.append(wklad_wlasny[-1])

        # aktualizacja docelowej wartości FIRE uwzględniając inflację
        docelowa_wartosc_fire_realna *= (1 + miesieczna_inflacja)

        if saldo >= docelowa_wartosc_fire_realna and fire_miesiac is None:
            fire_miesiac = miesiac

        # liczymy roczne wyniki co 12 miesięcy
        if miesiac % 12 == 0:
            roczny_zysk = saldo - wklad_wlasny[miesiac]
            # jeśli wkład własny w tym momencie jest zerowy, unikamy dzielenia przez zero
            if wklad_wlasny[miesiac] != 0:
                roczny_zysk_procentowy = (roczny_zysk / wklad_wlasny[miesiac]) * 100
            else:
                roczny_zysk_procentowy = 0.0

            zysk_roczny.append(roczny_zysk)
            zysk_roczny_procentowy.append(roczny_zysk_procentowy)

            # procentowy wzrost portfela względem początku roku
            if len(wartosc_portfela) > 12:
                poczatek_roku = wartosc_portfela[miesiac - 12]
                koniec_roku = wartosc_portfela[miesiac]
                if poczatek_roku > 0:
                    procentowy_wzrost = ((koniec_roku - poczatek_roku) / poczatek_roku) * 100
                else:
                    procentowy_wzrost = 0.0
                procentowy_wzrost_portfela.append(procentowy_wzrost)
            else:
                procentowy_wzrost_portfela.append(0.0)

    # Oblicz różnice między aktualnym wzrostem a poprzednim (zwrot jako różnica)
    zwrot_roznica = []
    for i in range(len(procentowy_wzrost_portfela)):
        if i == 0:
            zwrot_roznica.append(0.0)  # brak poprzedniego roku -> 0
        else:
            diff = procentowy_wzrost_portfela[i] - procentowy_wzrost_portfela[i - 1]
            zwrot_roznica.append(diff)

    return wartosc_portfela, wklad_wlasny, zysk_roczny, zysk_roczny_procentowy, procentowy_wzrost_portfela, zwrot_roznica, fire_miesiac

def format_number(number):
    return f"{number:,.2f}".replace(",", " ").replace(".", ",")

def wyswietl_wyniki(zysk_roczny, zysk_roczny_procentowy, zwrot_roznica, wklad_wlasny, wartosc_portfela, fire_miesiac):
    finalna_wartosc_portfela = wartosc_portfela[-1]
    finalny_wklad_wlasny = wklad_wlasny[-1]
    finalny_zwrot = finalna_wartosc_portfela - finalny_wklad_wlasny

    # Nagłówek: usunięto kolumnę "Wzrost wartości portfela (%)"
    print(f"{'Rok':<5}{'Wkład własny (PLN)':<25}{'Zysk roczny (PLN)':<20}{'Zwrot (%)':<15}")
    for i in range(1, len(zysk_roczny) + 1):
        # indeks na koniec roku to i*12
        wk = wklad_wlasny[i * 12]
        zysk = zysk_roczny[i - 1]
        zw = zwrot_roznica[i - 1] if i - 1 < len(zwrot_roznica) else 0.0
        print(f"{i:<5}{format_number(wk):<25}{format_number(zysk):<20}{zw:<15.2f}")

    print(f"\nFinalna wartość portfela: {format_number(finalna_wartosc_portfela)} PLN")
    print(f"Całkowity wkład własny: {format_number(finalny_wklad_wlasny)} PLN")
    print(f"Łączny zysk z inwestycji: {format_number(finalny_zwrot)} PLN")

    if fire_miesiac:
        rok_fire = fire_miesiac // 12
        miesiac_fire = fire_miesiac % 12
        print(f"Możesz przejść na FIRE w miesiącu {miesiac_fire} roku {rok_fire} od dziś.")
    else:
        print("Nie osiągnięto FIRE w podanym okresie.")

def stworz_wykresy(wartosc_portfela, wklad_wlasny, zysk_roczny, zwrot_roznica):
    plt.figure(figsize=(14, 4))

    # Wykres 1: Wartość portfela i wkład własny
    plt.subplot(1, 3, 1)
    plt.plot(wartosc_portfela, label='Wartość portfela')
    plt.plot(wklad_wlasny, label='Wkład własny', linestyle='--')
    plt.xlabel('Miesiące')
    plt.ylabel('Wartość w PLN')
    plt.title('Wartość portfela w czasie')
    plt.legend()
    plt.grid(True)

    # Wykres 2: Roczne zyski
    plt.subplot(1, 3, 2)
    plt.bar(range(1, len(zysk_roczny) + 1), zysk_roczny, alpha=0.7)
    plt.xlabel('Rok')
    plt.ylabel('Zysk roczny (PLN)')
    plt.title('Roczne zyski portfela')

    # Wykres 3: Zwrot jako różnica między aktualnym a poprzednim rocznym wzrostem (procentowo)
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(zwrot_roznica) + 1), zwrot_roznica, marker='o')
    plt.xlabel('Rok')
    plt.ylabel('Zwrot - różnica (%)')
    plt.title('Roczny zwrot (różnica wzrostów)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    roczny_zwrot = 0.20
    wklad_miesieczny = 4000
    ilosc_lat = 30
    roczne_wydatki = 72000
    inflacja = 0.03
    miesiace_bez_wplatow = 70  # przykład: przestajemy wpłacać po 1 miesiącu

    (wartosc_portfela,
     wklad_wlasny,
     zysk_roczny,
     zysk_roczny_procentowy,
     procentowy_wzrost_portfela,
     zwrot_roznica,
     fire_miesiac) = oblicz_portfel(
        roczny_zwrot, wklad_miesieczny, ilosc_lat, roczne_wydatki, inflacja, miesiace_bez_wplatow
    )

    wyswietl_wyniki(zysk_roczny, zysk_roczny_procentowy, zwrot_roznica, wklad_wlasny, wartosc_portfela, fire_miesiac)
    stworz_wykresy(wartosc_portfela, wklad_wlasny, zysk_roczny, zwrot_roznica)

if __name__ == "__main__":
    main()
