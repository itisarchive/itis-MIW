% labirynt.pl

labyrinth_size(5, 5).
:- dynamic labyrinth_size/2.

labirynt(W, H) :-
    retractall(labyrinth_size(_, _)),
    assertz(labyrinth_size(W, H)),
    start.

:- dynamic cell/3.
:- dynamic visited/2.
:- dynamic player_pos/2.
:- dynamic dragon_pos/2.
:- dynamic princess_pos/2.
:- dynamic score/1.
:- dynamic got_princess/0.

initialize_labyrinth :-
    labyrinth_size(Width, Height),
    forall(
        (between(1, Width, X), between(1, Height, Y)),
        assertz(cell(X, Y, [north, south, east, west]))
    ).

generate_maze :-
    labyrinth_size(Width, Height),
    random_between(1, Width, StartX),
    random_between(1, Height, StartY),
    generate_maze(StartX, StartY).

generate_maze(X, Y) :-
    assertz(visited(X, Y)),
    findall(
        (NX, NY, Direction),
        neighbor(X, Y, NX, NY, Direction),
        Neighbors
    ),
    random_permutation(Neighbors, ShuffledNeighbors),
    visit_neighbors(X, Y, ShuffledNeighbors).

visit_neighbors(_, _, []).
visit_neighbors(X, Y, [(NX, NY, Direction)|Rest]) :-
    (   visited(NX, NY)
    ->  visit_neighbors(X, Y, Rest)
    ;   remove_wall(X, Y, NX, NY, Direction),
        generate_maze(NX, NY),
        visit_neighbors(X, Y, Rest)
    ).

neighbor(X, Y, NX, Y, east) :-
    labyrinth_size(Width, _),
    X < Width,
    NX is X + 1.
neighbor(X, Y, NX, Y, west) :-
    X > 1,
    NX is X - 1.
neighbor(X, Y, X, NY, south) :-
    labyrinth_size(_, Height),
    Y < Height,
    NY is Y + 1.
neighbor(X, Y, X, NY, north) :-
    Y > 1,
    NY is Y - 1.

remove_wall(X1, Y1, X2, Y2, Direction) :-
    retract(cell(X1, Y1, Walls1)),
    delete(Walls1, Direction, NewWalls1),
    assertz(cell(X1, Y1, NewWalls1)),
    opposite(Direction, Opposite),
    retract(cell(X2, Y2, Walls2)),
    delete(Walls2, Opposite, NewWalls2),
    assertz(cell(X2, Y2, NewWalls2)).

opposite(north, south).
opposite(south, north).
opposite(east, west).
opposite(west, east).

add_entrances :-
    labyrinth_size(Width, Height),
    retract(cell(1, 1, Walls1)),
    delete(Walls1, north, NewWalls1),
    assertz(cell(1, 1, NewWalls1)),
    retract(cell(Width, Height, Walls2)),
    delete(Walls2, east, NewWalls2),
    assertz(cell(Width, Height, NewWalls2)).

init_player :-
    retractall(player_pos(_, _)),
    assertz(player_pos(1, 1)).

init_entities :-
    labyrinth_size(Width, Height),
    place_dragon(Width, Height),
    place_princess(Width, Height).

place_dragon(Width, Height) :-
    repeat,
    random_between(1, Width, Dx),
    random_between(1, Height, Dy),
    ( (Dx \= 1; Dy \= 1),
      (Dx \= Width; Dy \= Height) ->
        retractall(dragon_pos(_, _)),
        assertz(dragon_pos(Dx, Dy)),
        !
    ; 
        fail
    ).

place_princess(Width, Height) :-
    dragon_pos(Dx, Dy),
    repeat,
    random_between(1, Width, Px),
    random_between(1, Height, Py),
    ( (Px \= 1; Py \= 1),
      (Px \= Dx; Py \= Dy),
      (Px \= Width; Py \= Height) ->
        retractall(princess_pos(_, _)),
        assertz(princess_pos(Px, Py)),
        !
    ;
        fail
    ).

display_row(Y, Width) :-
    display_horizontal_walls(Y, Width),
    display_vertical_walls(Y, Width).

display_horizontal_walls(Y, Width) :-
    forall(between(1, Width, X), (
        cell(X, Y, Walls),
        (member(north, Walls) -> write('+---') ; write('+   '))
    )),
    write('+'), nl.

display_vertical_walls(Y, Width) :-
    forall(between(1, Width, X), (
        cell(X, Y, Walls),
        (member(west, Walls) -> write('|') ; write(' ')),
        ( player_pos(Px, Py),
          X == Px, Y == Py -> write(' X ')
        ; write('   ')
        )
    )),
    cell(Width, Y, Walls),
    (member(east, Walls) -> write('|') ; write(' ')),
    nl.

display_bottom_walls(Y, Width) :-
    forall(between(1, Width, X), (
        cell(X, Y, Walls),
        (member(south, Walls) -> write('+---') ; write('+   '))
    )),
    write('+'), nl.

can_move(X, Y, north, X, NY) :-
    Y > 1,
    NY is Y - 1,
    cell(X, Y, Walls),
    \+ member(north, Walls).

can_move(X, Y, south, X, NY) :-
    labyrinth_size(_, Height),
    Y < Height,
    NY is Y + 1,
    cell(X, Y, Walls),
    \+ member(south, Walls).

can_move(X, Y, east, NX, Y) :-
    labyrinth_size(Width, _),
    X < Width,
    NX is X + 1,
    cell(X, Y, Walls),
    \+ member(east, Walls).

can_move(X, Y, west, NX, Y) :-
    X > 1,
    NX is X - 1,
    cell(X, Y, Walls),
    \+ member(west, Walls).

direction(polnoc, north).
direction(poludnie, south).
direction(wschod, east).
direction(zachod, west).
direction(n, north).
direction(s, south).
direction(e, east).
direction(w, west).

mapa :-
    labyrinth_size(Width, Height),
    forall(between(1, Height, Y), display_row(Y, Width)),
    display_bottom_walls(Height, Width).

increase_score(Value) :-
    retract(score(Old)),
    NewScore is Old + Value,
    assertz(score(NewScore)).

attempt_move(DirectionEnglish, Cost, Mode) :-
    player_pos(X, Y),
    (   can_move(X, Y, DirectionEnglish, NX, NY)
    ->  retract(player_pos(X, Y)),
        assertz(player_pos(NX, NY)),
        increase_score(Cost),
        check_cell_entry(NX, NY, Mode)
    ;   format("Nie mozesz tam pojsc, jest sciana!~n", [])
    ).

check_cell_entry(NX, NY, Mode) :-
    ( dragon_pos(Dx, Dy), NX == Dx, NY == Dy, Mode == normal ->
        format("Wchodzisz do komnaty... Tu jest smok! Zostajesz pozarty... Koniec gry!~n", []),
        show_final_score,
        halt
    ; princess_pos(Px, Py), NX == Px, NY == Py, \+ got_princess ->
        assertz(got_princess),
        format("Znalazles ksiezniczke! Teraz mozesz szukac wyjscia.~n", [])
    ; true
    ).

sprawdz(DirectionPolish) :-
    (   direction(DirectionPolish, DirectionEnglish)
    ->  increase_score(1),
        player_pos(X, Y),
        (   neighbor(X, Y, NX, NY, DirectionEnglish)
        ->  (dragon_pos(Dx, Dy), NX == Dx, NY == Dy ->
                format("Slyszysz warczenie smoka!~n", [])
            ; princess_pos(Px, Py), NX == Px, NY == Py ->
                format("Slyszysz ciche nawolywanie, to moze byc ksiezniczka!~n", [])
            ; format("Cisza, nic tam nie ma...~n", [])
            )
        ;   format("Nie ma tam komnaty do sprawdzenia!~n", [])
        )
    ;   format("Nieznany kierunek: ~w~n", [DirectionPolish])
    ).

idz(DirectionPolish) :-
    (   direction(DirectionPolish, DirectionEnglish)
    ->  format("Idziesz w kierunku ~w...~n", [DirectionPolish]),
        attempt_move(DirectionEnglish, 1, normal)
    ;   format("Nierozpoznany kierunek: ~w~n", [DirectionPolish])
    ).

zakradnij(DirectionPolish) :-
    (   direction(DirectionPolish, DirectionEnglish)
    ->  attempt_move(DirectionEnglish, 3, silent),
        format("Zakradasz sie cicho w kierunku ~w...~n", [DirectionPolish])
    ;   format("Nierozpoznany kierunek: ~w~n", [DirectionPolish])
    ).

ucieknij :-
    labyrinth_size(Width, Height),
    player_pos(X, Y),
    (X == Width, Y == Height ->
        ( got_princess ->
            format("Udalo ci sie uciec z ksiezniczka! Gratulacje!~n", []),
            show_final_score,
            halt
        ; format("To wyjscie, ale nie masz ksiezniczki!~n", [])
        )
    ; format("To nie jest wyjscie, nie mozesz uciekac!~n", [])
    ).

show_final_score :-
    score(S),
    format("Twoj koncowy wynik: ~d~n", [S]).

start :-
    retractall(cell(_, _, _)),
    retractall(visited(_, _)),
    retractall(player_pos(_, _)),
    retractall(dragon_pos(_, _)),
    retractall(princess_pos(_, _)),
    retractall(score(_)),
    retractall(got_princess),
    assertz(score(0)),
    initialize_labyrinth,
    generate_maze,
    add_entrances,
    init_player,
    init_entities,
    pomoc,
    mapa.

pomoc :-
    write("\nWitamy w swiecie Faerunu!\n"),
    write("\nW mistycznych krainach Faerunu znajduje sie ukryta wyspa Evermeet, zamieszkala przez enigmatyczne elfy. Pod ta spokojna wyspa rozciaga sie zdradziecki loch, pelen mrocznych tajemnic i niebezpieczenstw.\n"),
    write("Okrutny Lord Vorthar, tyran znany ze swojej czarnej magii i bezlitosnych sposobow, przetrzymuje piekna ksiezniczke Rozalie w glebinach tego lochu.\n"),
    write("Jako odwazny bohater, twoim przeznaczeniem jest wkroczyc do lochu, uratowac ksiezniczke Rozalie i wspolnie uciec na wolnosc.\n\n"),

    write("Masz mape lochu, ktora pomoze ci w nawigacji, ale droga nie bedzie latwa.\n"),
    write("Uzyj swojej madrosci i odwagi, aby poruszac sie po labiryncie, unikac smoka strzegacego glebin i odnalezc ksiezniczke.\n\n"),

    write("Oto dostepne polecenia:\n\n"),

    write("1. idz(DIRECTION).\n"),
    write("   Porusz sie w podanym kierunku (polnoc, poludnie, wschod, zachod) lub (n, s, w, e).\n"),
    write("2. zakradnij(DIRECTION).\n"),
    write("   Zakradnij sie cicho w podanym kierunku, unikajac natychmiastowego niebezpieczenstwa, ale kosztem wiekszej energii.\n"),
    write("3. sprawdz(DIRECTION).\n"),
    write("   Sprawdz sasiednia komnate w podanym kierunku, aby dowiedziec sie o zagrozeniach lub wskazowkach dotyczacych tego, co sie tam znajduje.\n"),
    write("4. mapa.\n"),
    write("   Wyswietl aktualna mape lochu, pokazujaca twoja pozycje.\n"),
    write("5. ucieknij.\n"),
    write("   Sprobuj uciec z lochu. Mozesz uciec tylko z ostatniej komnaty, jesli uratowales ksiezniczke Rozalie.\n"),
    write("6. start.\n"),
    write("   Rozpocznij nowa rozgrywke.\n"),

    write("Twoja przygoda zaczyna sie teraz. Badz odwazny i madry, bohaterze Faerunu!\n\n").
